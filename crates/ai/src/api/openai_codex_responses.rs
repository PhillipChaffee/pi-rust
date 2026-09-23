//! The OpenAI Codex Responses wire API, ported from
//! `packages/ai/src/api/openai-codex-responses.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The ChatGPT/Codex client collapses onto the [`crate::http::HttpClient`]
//! seam for the SSE transport and onto the [`crate::http::websocket`]
//! transport for the WebSocket one: upstream's `getWebSocketConstructor`
//! ambient `globalThis.WebSocket` (and its Bun-only proxy subclass) ports to
//! [`crate::http::default_websocket_transport`], with the module-local
//! [`set_websocket_transport`] standing in for `vi.stubGlobal("WebSocket",
//! ...)` in the ported tests. Upstream's `options.fetch` injection rides the
//! [`TransportOptions::http_client`] seam like the sibling wire APIs.
//!
//! Porting restatements:
//!
//! - `sanitizeSurrogates` disappears statically: a Rust [`String`] cannot
//!   hold the unpaired surrogates it stripped, and `serde_json` rejects
//!   lone-surrogate escapes when reading the wire.
//! - The event seam: upstream feeds `processResponsesStream` a mapped async
//!   iterable (`parseSSE`/`parseWebSocket` through `mapCodexEvents`); the
//!   Rust shared processor consumes an [`HttpResponse`], so the mapped events
//!   ride a synthetic SSE body — each mapped event re-framed as one
//!   `data:` line — and the shared Responses state machine consumes them
//!   untouched. Wire `error` and `response.failed` events never reach it: the
//!   mapper raises them as the module's typed failures, the wire's
//!   `error.code` riding alongside for the retry classification.
//! - `streamSimple`'s synchronous auth throw becomes the setup-error stream;
//!   `stream`'s async throws surface as its `error` event. The catch block's
//!   `formatProviderError(normalizeProviderError(error))` composition is an
//!   identity on the plain thrown Errors codex raises, so the message rides
//!   verbatim, and the catch's scratch-field cleanup disappears statically
//!   (the shared processor never persists the streamed scratch buffers).
//! - `compressRequestBodyZstd` ports to a module-local zstd frame writer: the
//!   pinned stack has no zstd compressor crate, so the SSE body rides a
//!   spec-valid stored-block zstd frame — the same `content-encoding: zstd`
//!   the backend decodes, without a compression ratio. Upstream's
//!   compression-unavailable fallback (uncompressed JSON, no header) is
//!   unreachable. The WebSocket frame stays uncompressed JSON, matching the
//!   official Codex client upstream matches.
//! - `normalizeTimeoutMs`'s invalid-timeout throw is statically unreachable:
//!   the Rust option is a `u64` and cannot carry the negative or non-finite
//!   values it rejected. `Math.floor` rides the integer type.
//! - The per-attempt combined abort signal (`options.signal` +
//!   `AbortSignal.timeout(httpTimeoutMs)`) collapses onto
//!   [`HttpRequest::timeout_ms`] plus the caller token: the seam's total
//!   request timeout covers headers and body alike, so a mid-body expiry
//!   surfaces as a timeout error where the TS signal would abort the read.
//!   The retry loop's fake-timer scheduling runs on `tokio::time::sleep`, so
//!   tests drive it with the paused clock the way the fake-timer suites drive
//!   the rest of the crate; the HTTP-date `Retry-After` distance reads the
//!   process clock through [`crate::auth::resolve::now_ms`].
//! - The SSE error parser reads the raw body alone: the seam response carries
//!   no `statusText`, so upstream's `raw || statusText || "Request failed"`
//!   fallback chain loses its middle step.
//! - The retry and error branches are codex's own: `isRetryableError`'s
//!   status/text policy, the `Retry-After` parsing including the HTTP-date
//!   form, the exact `1000 * 2^n` backoff, the `RetryDelayExceededError`
//!   wording, and the usage-limit catch guard. They do not route through
//!   [`crate::utils::provider_retry::retry_provider_request`], whose
//!   SDK-shaped policy (x-should-retry, jittered backoff, `. {message}`
//!   delay-cap wording, no HTTP-date parsing) differs on every axis, and the
//!   sibling's `sdk_error_message` composition is likewise absent — upstream
//!   throws `parseErrorResponse`'s friendly message, which the port
//!   reproduces verbatim.
//! - The WebSocket session cache keeps upstream's process-global shape: one
//!   connection per `(sessionId, accountId)` pair, idle-expired after five
//!   minutes, replaced at the backend's fifty-five-minute connection age
//!   limit, and registered as a session resource so
//!   [`crate::session_resources`] teardown closes it. The WebSocket seam
//!   carries no `readyState`, so a cached connection's liveness is discovered
//!   on use — a connection the peer dropped between requests surfaces as a
//!   transport failure, which the SSE-fallback machine absorbs the same way
//!   upstream's reusability check does. A stream dropped mid-poll returns the
//!   checked-out connection through its slot; the release path closes it.
//! - `connectWebSocket`'s `delete wsHeaders["OpenAI-Beta"]` misses the
//!   lowercased name the `Headers`-to-record round-trip produces, so the
//!   `responses_websockets` beta header rides the handshake; the port sends
//!   the handshake header names lowercased the same way.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use futures_util::stream::unfold;
use regex::Regex;
use serde_json::{Map, Value, json};
use tokio_util::sync::CancellationToken;

use crate::api::constrained_sampling::create_grammar_tool_input_properties;
use crate::api::openai_prompt_cache::clamp_openai_prompt_cache_key;
use crate::api::openai_responses::{mapped_effort, pi_thinking_level, thinking_model_level};
use crate::api::openai_responses_shared::{
    ConvertResponsesMessagesOptions, ConvertResponsesToolsOptions, OpenAiResponsesStreamOptions,
    ResolveServiceTier, ResponsesDeferredToolsMode, convert_responses_messages,
    convert_responses_tools, pricing_hook, process_responses_stream,
};
use crate::api::request_seam::{
    fire_response_hook, read_body_text, setup_error_stream, spawned_stream,
};
use crate::api::simple_options::build_base_options;
use crate::auth::resolve::now_ms;
use crate::http::client::{HttpByteStream, HttpError, HttpMethod, HttpRequest, HttpResponse};
use crate::http::sse::SseStream;
use crate::http::websocket::{
    WebSocketConnection, WebSocketError, WebSocketMessage, WebSocketRequest, WebSocketTransport,
};
use crate::models::clamp_thinking_level;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, CacheRetention, Context, Message, Model,
    ModelThinkingLevel, ProviderEnv, ProviderHeaders, SimpleStreamOptions, StopReason,
    StreamOptions, Tool, ToolChoice, Transport, TransportOptions,
};
use crate::utils::deferred_tools::split_deferred_tools;
use crate::utils::diagnostics::{
    append_assistant_message_diagnostic, create_assistant_message_diagnostic, format_thrown_value,
};
use crate::utils::event_stream::AssistantMessageEventStream;
use crate::utils::pi_user_agent::get_pi_user_agent;
use crate::utils::provider_retry::DEFAULT_MAX_RETRY_DELAY_MS;
use crate::utils::uuid::uuidv7;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The ChatGPT backend the Codex wire API talks to when the model carries no
/// base URL, upstream's `DEFAULT_CODEX_BASE_URL`.
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
/// The JWT claim object the ChatGPT auth token carries the account id under,
/// upstream's `JWT_CLAIM_PATH`.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
/// Retries a Codex stream starts with when the options carry none, upstream's
/// `DEFAULT_MAX_RETRIES`.
const DEFAULT_MAX_RETRIES: u32 = 0;
/// The first backoff step of the SSE retry schedule, upstream's
/// `BASE_DELAY_MS`; each later attempt doubles it.
const BASE_DELAY_MS: u64 = 1_000;
/// The WebSocket beta feature string the handshake sends, upstream's
/// `OPENAI_BETA_RESPONSES_WEBSOCKETS`.
const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
/// How long an idle cached WebSocket stays connected, upstream's
/// `SESSION_WEBSOCKET_CACHE_TTL_MS`.
const SESSION_WEBSOCKET_CACHE_TTL_MS: u64 = 5 * 60 * 1000;
/// The backend connection age a cached WebSocket is replaced before, upstream's
/// `SESSION_WEBSOCKET_MAX_AGE_MS`.
const SESSION_WEBSOCKET_MAX_AGE_MS: i64 = 55 * 60 * 1000;
/// The close code the backend uses when a WebSocket frame exceeds its size
/// limit, upstream's `WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE`.
const WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE: u16 = 1009;
/// The wire error code a full backend connection pool reports, upstream's
/// `WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE`; one fresh reconnect is spent on
/// it before any fallback.
const WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE: &str = "websocket_connection_limit_reached";
/// The wire error code a cached-connection continuation fails with when the
/// backend forgot `previous_response_id`; the stream retries once on a full
/// input, upstream's `PREVIOUS_RESPONSE_NOT_FOUND_CODE`.
const PREVIOUS_RESPONSE_NOT_FOUND_CODE: &str = "previous_response_not_found";

/// The providers whose tool-call item ids carry OpenAI Responses pairing
/// history, upstream's `CODEX_TOOL_CALL_PROVIDERS`.
fn codex_tool_call_providers() -> BTreeSet<String> {
    ["openai", "openai-codex", "opencode"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// The response statuses the Codex terminal events carry, upstream's
/// `CODEX_RESPONSE_STATUSES`; anything else normalizes to absent.
fn codex_response_status(status: &str) -> Option<&str> {
    match status {
        "completed" | "incomplete" | "failed" | "cancelled" | "queued" | "in_progress" => {
            Some(status)
        }
        _ => None,
    }
}

/// The transport as the wire spells it, upstream's `Transport` union.
const fn transport_wire(transport: Transport) -> &'static str {
    match transport {
        Transport::Sse => "sse",
        Transport::Websocket => "websocket",
        Transport::WebsocketCached => "websocket-cached",
        Transport::Auto => "auto",
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The Codex `tool_choice` request value, upstream's
/// `"auto" | "none" | "required"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodexToolChoice {
    /// `"auto"`: the model chooses.
    Auto,
    /// `"none"`: no tool calls.
    None,
    /// `"required"`: the model must call a tool.
    Required,
}

impl CodexToolChoice {
    /// The value as the wire spells it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Required => "required",
        }
    }
}

/// The OpenAI Codex Responses options, upstream's
/// `OpenAICodexResponsesOptions extends StreamOptions`.
///
/// The base fields plus the reasoning, service-tier, text-verbosity, and
/// tool-choice extras the ChatGPT backend reads. This is the one
/// Responses-family adapter that reads the `transport` and
/// `websocketConnectTimeoutMs` base fields.
#[derive(Clone, Debug, Default)]
pub struct OpenAiCodexResponsesOptions {
    /// The transport seam: the request's HTTP client, cancellation token,
    /// and lifecycle callbacks.
    pub transport_options: TransportOptions,
    /// The ChatGPT OAuth token; the `chatgpt_account_id` claim is read from
    /// it. Required — the setup fails without one.
    pub api_key: Option<String>,
    /// Explicit parent context for telemetry produced by this request.
    pub telemetry_context: Option<pi_telemetry::TelemetryHandle>,
    /// Provider-scoped environment values. The WebSocket transport reads none
    /// of them: the Bun proxy resolution upstream pairs with it stays out of
    /// the pinned transport.
    pub env: Option<ProviderEnv>,
    /// Custom HTTP headers merged over the model's; `None` values suppress a
    /// default header. The forced `Authorization`, `chatgpt-account-id`,
    /// `originator`, and `User-Agent` headers override these.
    pub headers: Option<ProviderHeaders>,
    /// HTTP request timeout in milliseconds. Doubles as the WebSocket
    /// stream's idle timeout: the wire's `timeoutMs` caps the SSE header wait
    /// and the WebSocket read idleness, not the connect handshake.
    pub timeout_ms: Option<u64>,
    /// Maximum retry attempts for the SSE request. Default: 0.
    pub max_retries: Option<u32>,
    /// Maximum delay in milliseconds to wait for a server-requested retry.
    /// Default: 60000 (60 seconds). `0` disables the cap.
    pub max_retry_delay_ms: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum output tokens. The adapter does not read it: the Codex
    /// endpoint carries no max-output request field.
    pub max_tokens: Option<u64>,
    /// Preferred transport, upstream's `"sse" | "websocket" |
    /// "websocket-cached" | "auto"`. Default: `"auto"` — WebSocket first,
    /// falling back to SSE when the connection fails before output starts.
    pub transport: Option<Transport>,
    /// Prompt cache retention preference. Default: `"short"`. `"none"` drops
    /// the session-affinity headers, the `prompt_cache_key`, and the
    /// WebSocket cache.
    pub cache_retention: Option<CacheRetention>,
    /// Session identifier for the cache-affinity headers, the
    /// `prompt_cache_key` request field, and the WebSocket session cache.
    pub session_id: Option<String>,
    /// WebSocket connect-handshake timeout in milliseconds. Default: 15000
    /// (15 seconds). A zero rides the transport default.
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Optional request metadata; the adapter does not read it.
    pub metadata: Option<BTreeMap<String, Value>>,
    /// The `reasoning.effort` request value, upstream's
    /// `"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"` —
    /// the off state is the wire's `"none"` effort, resolved through the
    /// model's `thinkingLevelMap.off`.
    pub reasoning_effort: Option<ModelThinkingLevel>,
    /// The `reasoning.summary` request value, upstream's
    /// `"auto" | "concise" | "detailed" | "off" | "on" | null`; absent and
    /// null both default to `"auto"`.
    pub reasoning_summary: Option<String>,
    /// The request's `service_tier` (`"auto"`, `"default"`, `"flex"`, or
    /// `"priority"`), priced into the final usage.
    pub service_tier: Option<String>,
    /// The `text.verbosity` request value, upstream's
    /// `"low" | "medium" | "high"`. Default: `"low"`.
    pub text_verbosity: Option<String>,
    /// The Responses `tool_choice` value. Default: `"auto"`.
    pub tool_choice: Option<CodexToolChoice>,
}

crate::api::adapter_belt::impl_stream_options_from!(OpenAiCodexResponsesOptions from options {
    transport: options.transport,
    websocket_connect_timeout_ms: options.websocket_connect_timeout_ms,
    reasoning_effort: None,
    reasoning_summary: None,
    service_tier: None,
    text_verbosity: None,
    tool_choice: None,
});
// ---------------------------------------------------------------------------
// Stream failures
// ---------------------------------------------------------------------------

/// The failure class of a mapped codex stream, upstream's thrown-error
/// taxonomy: `CodexApiError`, `CodexProtocolError`, and every other `Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamFailureKind {
    /// A wire `error`/`response.failed` event, upstream's `CodexApiError`.
    Api,
    /// A frame that does not parse, upstream's `CodexProtocolError`.
    Protocol,
    /// Every other thrown `Error` — aborts, timeouts, transport failures.
    Plain,
}

/// Why a codex stream failed, carrying the composed `errorMessage` and the
/// classification the WebSocket fallback machine retries on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StreamFailure {
    /// The failure description, the catch block's `errorMessage`.
    message: String,
    /// The thrown-value classification.
    kind: StreamFailureKind,
    /// The wire's `error.code` when the failure carried one.
    code: Option<String>,
}

impl StreamFailure {
    fn plain(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: StreamFailureKind::Plain,
            code: None,
        }
    }

    fn api(message: impl Into<String>, code: Option<String>) -> Self {
        Self {
            message: message.into(),
            kind: StreamFailureKind::Api,
            code,
        }
    }

    fn protocol(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: StreamFailureKind::Protocol,
            code: None,
        }
    }

    /// Whether the failure is a wire-level API rejection upstream does not
    /// fall back on, upstream's `isCodexNonTransportError`.
    const fn is_non_transport(&self) -> bool {
        matches!(
            self.kind,
            StreamFailureKind::Api | StreamFailureKind::Protocol
        )
    }

    /// Whether the backend reported a full connection pool, upstream's
    /// `isWebSocketConnectionLimitReachedError`.
    fn is_websocket_connection_limit(&self) -> bool {
        self.kind == StreamFailureKind::Api
            && self.code.as_deref() == Some(WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE)
    }

    /// Whether the backend forgot a cached continuation, upstream's
    /// `isPreviousResponseNotFoundError`.
    fn is_previous_response_not_found(&self) -> bool {
        self.kind == StreamFailureKind::Api
            && self.code.as_deref() == Some(PREVIOUS_RESPONSE_NOT_FOUND_CODE)
    }
}

impl std::fmt::Display for StreamFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StreamFailure {}

// ---------------------------------------------------------------------------
// Retry classification and scheduling
// ---------------------------------------------------------------------------

/// Subscription and account limits that look like throttles but must never
/// retry, upstream's `isTerminalRateLimitError` pattern.
#[expect(
    clippy::expect_used,
    reason = "the joined pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static TERMINAL_RATE_LIMIT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "(?i)GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota|out of budget|quota exceeded|billing",
    )
    .expect("the terminal rate-limit pattern is a valid regex")
});

/// Transient provider and transport wording worth retrying, upstream's
/// `isRetryableError` fallback pattern.
#[expect(
    clippy::expect_used,
    reason = "the joined pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static RETRYABLE_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "(?i)rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
    )
    .expect("the retryable pattern is a valid regex")
});

/// The error codes whose usage-limit wording the friendly-message parser
/// recognizes, upstream's `parseErrorResponse` code pattern.
#[expect(
    clippy::expect_used,
    reason = "the joined pattern is a compile-time constant; a failure is a programming error, not a runtime condition"
)]
static USAGE_LIMIT_CODE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new("(?i)usage_limit_reached|usage_not_included|rate_limit_exceeded")
        .expect("the usage-limit code pattern is a valid regex")
});

/// Whether the failure text names a spent subscription the retry schedule
/// must honor rather than retry, upstream's `isTerminalRateLimitError`.
fn is_terminal_rate_limit_error(error_text: &str) -> bool {
    TERMINAL_RATE_LIMIT_PATTERN.is_match(error_text)
}

/// Whether a failed response is worth a retry, upstream's `isRetryableError`:
/// non-terminal 429s and the transient 5xx set, else the error-text pattern.
fn is_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 && is_terminal_rate_limit_error(error_text) {
        return false;
    }
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    RETRYABLE_ERROR_PATTERN.is_match(error_text)
}

/// Read a header case-insensitively, like `Headers.get`.
fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(name_candidate, _)| name_candidate.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

/// The server-requested delay, upstream's `getRetryAfterDelayMs`:
/// `retry-after-ms` milliseconds, then `retry-after` seconds, then the
/// HTTP-date form as the wall-clock distance to it.
fn get_retry_after_delay_ms(headers: &[(String, String)]) -> Option<f64> {
    if let Some(retry_after_ms) = header_value(headers, "retry-after-ms")
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|delay_ms| delay_ms.is_finite())
    {
        return Some(retry_after_ms.max(0.0));
    }

    let retry_after = header_value(headers, "retry-after")?;
    let retry_after = retry_after.trim();
    if retry_after.is_empty() {
        return None;
    }
    if let Ok(seconds) = retry_after.parse::<f64>()
        && seconds.is_finite()
    {
        return Some((seconds * 1000.0).max(0.0));
    }
    parse_http_date(retry_after).map(|when| {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the wall-clock millisecond distances enter the JS number math the timers read"
        )]
        let delay_ms = (when - now_ms()) as f64;
        delay_ms.max(0.0)
    })
}

/// The wall-clock milliseconds of an IMF-fixdate (`Mon, 15 Jun 2026
/// 12:00:45 GMT`), the form `toUTCString` produces that `Date.parse` reads;
/// other date spellings parse as absent, like a `NaN` `Date.parse` result.
fn parse_http_date(value: &str) -> Option<i64> {
    let mut fields = value.split_whitespace();
    let day = fields.nth(1)?;
    let month = fields.next()?;
    let year = fields.next()?;
    let time = fields.next()?;
    let month_index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|name| month.len() >= 3 && name.eq_ignore_ascii_case(&month[..3]))?;
    let month_number = u32::try_from(month_index)
        .ok()?
        .checked_add(1)
        .filter(|month| *month <= 12)?;
    let day: u32 = day.parse().ok()?;
    let year: i64 = year.parse().ok()?;
    let [hours, minutes, seconds] = time.split(':').collect::<Vec<_>>()[..] else {
        return None;
    };
    let hours: i64 = hours.parse().ok()?;
    let minutes: i64 = minutes.parse().ok()?;
    let seconds: i64 = seconds.parse().ok()?;
    if day == 0 || day > 31 || hours > 23 || minutes > 59 || seconds > 60 {
        return None;
    }
    Some(
        (days_from_civil(year, month_number, day) * 86_400
            + hours * 3_600
            + minutes * 60
            + seconds)
            * 1_000,
    )
}

/// Days from 1970-01-01 to a civil date, Howard Hinnant's `days_from_civil`.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let day_of_year = (153 * i64::from(if month > 2 { month - 3 } else { month + 9 }) + 2) / 5
        + i64::from(day)
        - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The delay cap check of a server-requested retry wait, upstream's
/// `validateRetryDelayMs`: a delay above the cap fails immediately with the
/// requested delay in the message so higher-level retry logic can surface it.
///
/// # Errors
/// The limit failure when the cap is enabled and the delay exceeds it.
fn validate_retry_delay_ms(
    delay_ms: f64,
    max_retry_delay_ms: Option<u64>,
) -> Result<f64, StreamFailure> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the delay cap enters the JS number math that compares server-requested delays"
    )]
    let max_delay_ms = max_retry_delay_ms.unwrap_or(DEFAULT_MAX_RETRY_DELAY_MS) as f64;
    if max_delay_ms > 0.0 && delay_ms > max_delay_ms {
        return Err(StreamFailure::plain(format!(
            "Server requested {}s retry delay (max: {}s)",
            (delay_ms / 1000.0).ceil(),
            (max_delay_ms / 1000.0).ceil()
        )));
    }
    Ok(delay_ms)
}

/// The backoff step of the SSE retry schedule, upstream's
/// `BASE_DELAY_MS * 2 ** attempt`.
const fn exponential_backoff_ms(attempt: u32) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "the integer schedule enters the JS number math the timers read"
    )]
    let delay_ms = BASE_DELAY_MS.saturating_mul(2_u64.saturating_pow(attempt)) as f64;
    delay_ms
}

/// Sleep a backoff step, stopping early when the caller aborts, upstream's
/// `sleep`. The sub-millisecond remainder truncates, matching the JS
/// scheduler the timeout ports.
///
/// # Errors
/// The abort failure when the signal cancels during the wait.
async fn sleep(delay_ms: f64, signal: &CancellationToken) -> Result<(), StreamFailure> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the delay is clamped to a non-negative millisecond wait before the cast"
    )]
    let delay = Duration::from_millis(delay_ms.max(0.0) as u64);
    tokio::select! {
        () = signal.cancelled() => Err(StreamFailure::plain("Request was aborted")),
        () = tokio::time::sleep(delay) => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Request compression
// ---------------------------------------------------------------------------

/// The largest content block a zstd block header can carry: the spec's
/// 128 KiB maximum.
const ZSTD_MAX_BLOCK_SIZE: usize = 131_072;

/// Wrap the JSON body in a zstd frame, upstream's `compressRequestBodyZstd`.
///
/// Porting restatement: node's `zstdCompressSync` at level 3 has no crate in
/// the pinned dependency set, so the body rides a spec-valid zstd frame of
/// raw (stored) blocks — every decoder, including the backend's
/// `Content-Encoding: zstd` handling and node's `zstdDecompressSync`, reads
/// it back byte-exact, without a compression ratio. The call cannot fail, so
/// upstream's compression-unavailable fallback never fires.
#[must_use]
fn compress_request_body_zstd(body_json: &str) -> Bytes {
    let content = body_json.as_bytes();
    let mut frame = Vec::with_capacity(content.len() + 32);
    // Magic number `0xFD2FB528` little-endian, then the frame header
    // descriptor: a 4-byte frame-content-size field under the single-segment
    // flag, so no window descriptor follows.
    frame.extend_from_slice(&[0x28, 0xB5, 0x2F, 0xFD]);
    frame.push(0xA0);
    frame.extend_from_slice(
        &u32::try_from(content.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    let mut offset = 0;
    loop {
        let remaining = content.len() - offset;
        let last_block = remaining <= ZSTD_MAX_BLOCK_SIZE;
        let block_size = remaining.min(ZSTD_MAX_BLOCK_SIZE);
        // The 3-byte block header: last-block bit 0, raw block type 0, size
        // in the upper 21 bits.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the block size is capped at 128 KiB, far below the 21-bit field"
        )]
        let header = u32::from(last_block) | ((block_size as u32) << 3);
        frame.extend_from_slice(&header.to_le_bytes()[..3]);
        frame.extend_from_slice(&content[offset..offset + block_size]);
        offset += block_size;
        if last_block {
            break;
        }
    }
    Bytes::from(frame)
}

// ---------------------------------------------------------------------------
// Auth and headers
// ---------------------------------------------------------------------------

/// The lenient base64 decode node's `atob` rides (`Buffer.from(data,
/// "base64")`): both alphabets, optional padding, invalid characters skipped.
fn decode_base64_segment(input: &str) -> Option<String> {
    let mut decoded: Vec<u8> = Vec::new();
    let mut accumulator: u32 = 0;
    let mut bits: u32 = 0;
    for byte in input.bytes() {
        let value = match byte {
            b'A'..=b'Z' => u32::from(byte - b'A'),
            b'a'..=b'z' => u32::from(byte - b'a') + 26,
            b'0'..=b'9' => u32::from(byte - b'0') + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => break,
            _ => continue,
        };
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the masked shift yields at most 8 bits, so the cast is exact"
            )]
            decoded.push((accumulator >> bits) as u8);
        }
    }
    String::from_utf8(decoded).ok()
}

/// The ChatGPT account id the auth token carries, upstream's
/// `extractAccountId`: the `chatgpt_account_id` claim of the token payload's
/// `https://api.openai.com/auth` object.
///
/// # Errors
/// The generic failure message when the token does not split into a
/// three-part JWT, the payload does not decode, or the claim is missing —
/// every upstream failure path collapses into the same message.
fn extract_account_id(token: &str) -> Result<String, StreamFailure> {
    let parts: Vec<&str> = token.split('.').collect();
    let account_id = (parts.len() == 3)
        .then(|| decode_base64_segment(parts[1]))
        .flatten()
        .and_then(|payload| serde_json::from_str::<Value>(&payload).ok())
        .and_then(|payload| {
            payload
                .get(JWT_CLAIM_PATH)?
                .get("chatgpt_account_id")?
                .as_str()
                .filter(|account_id| !account_id.is_empty())
                .map(str::to_owned)
        });
    account_id.ok_or_else(|| StreamFailure::plain("Failed to extract accountId from token"))
}

/// Drop every pair named `name` (case-insensitive), like `Headers.delete`.
fn delete_header(pairs: &mut Vec<(String, String)>, name: &str) {
    pairs.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
}

/// Push a header pair after dropping any earlier spelling, like `Headers.set`.
fn push_header(pairs: &mut Vec<(String, String)>, name: &str, value: &str) {
    delete_header(pairs, name);
    pairs.push((name.to_owned(), value.to_owned()));
}

/// Assemble the headers both Codex transports share, upstream's
/// `buildBaseCodexHeaders`: the model's headers, the caller's merged over
/// them (`None` suppresses), then the forced `Authorization`,
/// `chatgpt-account-id`, `originator: pi`, and pi user agent.
fn build_base_codex_headers(
    init_headers: Option<&BTreeMap<String, String>>,
    additional_headers: Option<&ProviderHeaders>,
    account_id: &str,
    token: &str,
) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = Vec::new();
    if let Some(init_headers) = init_headers {
        for (name, value) in init_headers {
            delete_header(&mut headers, name);
            headers.push((name.clone(), value.clone()));
        }
    }
    if let Some(additional_headers) = additional_headers {
        for (name, value) in additional_headers {
            match value {
                Some(value) => {
                    delete_header(&mut headers, name);
                    headers.push((name.clone(), value.clone()));
                }
                None => delete_header(&mut headers, name),
            }
        }
    }
    // The forced headers override both merges, like the trailing `set` calls.
    for (name, value) in [
        ("Authorization", format!("Bearer {token}")),
        ("chatgpt-account-id", account_id.to_owned()),
        ("originator", "pi".to_owned()),
        ("User-Agent", get_pi_user_agent()),
    ] {
        delete_header(&mut headers, name);
        headers.push((name.to_owned(), value));
    }
    headers
}

/// Assemble the SSE request headers, upstream's `buildSSEHeaders`: the beta
/// `responses=experimental` marker, the event-stream accept type, and the
/// session-affinity pair `session-id`/`x-client-request-id` when a session
/// id rides.
fn build_sse_headers(
    init_headers: Option<&BTreeMap<String, String>>,
    additional_headers: Option<&ProviderHeaders>,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = build_base_codex_headers(init_headers, additional_headers, account_id, token);
    for (name, value) in [
        ("OpenAI-Beta", "responses=experimental"),
        ("accept", "text/event-stream"),
        ("content-type", "application/json"),
    ] {
        push_header(&mut headers, name, value);
    }
    if let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty()) {
        for name in ["session-id", "x-client-request-id"] {
            push_header(&mut headers, name, session_id);
        }
    }
    headers
}

/// Assemble the WebSocket request headers, upstream's `buildWebSocketHeaders`:
/// the accept/content-type/beta headers the SSE set carries are dropped, the
/// `responses_websockets` beta value is set, and the request id rides the
/// session-affinity pair.
fn build_websocket_headers(
    init_headers: Option<&BTreeMap<String, String>>,
    additional_headers: Option<&ProviderHeaders>,
    account_id: &str,
    token: &str,
    request_id: &str,
) -> Vec<(String, String)> {
    let mut headers = build_base_codex_headers(init_headers, additional_headers, account_id, token);
    for name in ["accept", "content-type", "OpenAI-Beta", "openai-beta"] {
        delete_header(&mut headers, name);
    }
    push_header(
        &mut headers,
        "OpenAI-Beta",
        OPENAI_BETA_RESPONSES_WEBSOCKETS,
    );
    for (name, value) in [
        ("x-client-request-id", request_id),
        ("session-id", request_id),
    ] {
        push_header(&mut headers, name, value);
    }
    headers
}

/// The handshake headers a WebSocket connect sends, the record
/// `connectWebSocket` builds through `headersToRecord`: names lowercased
/// like `Headers.entries()` yields, later spellings collapsed. The beta
/// header rides — the record delete upstream runs misses the lowercased name.
///
/// # Errors
/// A header name outside the visible-ASCII range the handshake accepts.
fn websocket_handshake_record(
    headers: Vec<(String, String)>,
) -> Result<Vec<(String, String)>, StreamFailure> {
    let mut record: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        let lowered = name.to_ascii_lowercase();
        if lowered.bytes().any(|byte| !(0x21..=0x7e).contains(&byte)) {
            return Err(StreamFailure::plain(format!(
                "invalid websocket header name: {name}"
            )));
        }
        record.insert(lowered, value);
    }
    Ok(record.into_iter().collect())
}

// ---------------------------------------------------------------------------
// URLs
// ---------------------------------------------------------------------------

/// The Codex SSE endpoint, upstream's `resolveCodexUrl`: the model's base
/// URL (or the ChatGPT default), trailing slashes stripped, then the
/// `/codex/responses` path ensured.
#[must_use]
fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = base_url.filter(|base_url| !base_url.trim().is_empty());
    let normalized = raw.unwrap_or(DEFAULT_CODEX_BASE_URL);
    let normalized = normalized.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_owned()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

/// The Codex WebSocket endpoint, upstream's `resolveCodexWebSocketUrl`: the
/// SSE URL with its scheme swapped to `wss:`/`ws:`.
///
/// # Errors
/// A base URL that does not parse.
fn resolve_codex_websocket_url(base_url: Option<&str>) -> Result<String, StreamFailure> {
    let mut url = url::Url::parse(&resolve_codex_url(base_url))
        .map_err(|error| StreamFailure::plain(format!("Invalid URL: {error}")))?;
    match url.scheme() {
        "https" => {
            url.set_scheme("wss")
                .map_err(|()| StreamFailure::plain("Invalid URL: wss scheme rejected"))?;
        }
        "http" => {
            url.set_scheme("ws")
                .map_err(|()| StreamFailure::plain("Invalid URL: ws scheme rejected"))?;
        }
        _ => {}
    }
    Ok(url.to_string())
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Build the `response.create` request body, upstream's `buildRequestBody`.
/// The wire's `undefined` fields are absent, so only present fields insert,
/// in the wire's key order.
///
/// # Errors
/// The message- and tool-conversion rejections the input list hits.
fn build_request_body(
    model: &Model,
    context: &Context,
    options: &OpenAiCodexResponsesOptions,
    codex_session_id: Option<&str>,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Map<String, Value>, StreamFailure> {
    let compat = model.compat.as_ref();
    let supports_strict_mode = compat
        .and_then(|compat| compat.supports_strict_mode)
        .unwrap_or(true);
    let supports_openai_grammar_tools = compat
        .and_then(|compat| compat.supports_openai_grammar_tools)
        .unwrap_or(false);
    let deferred_tools_mode = if compat
        .and_then(|compat| compat.supports_additional_tools)
        .unwrap_or(false)
    {
        Some(ResponsesDeferredToolsMode::AdditionalTools)
    } else if compat
        .and_then(|compat| compat.supports_tool_search)
        .unwrap_or(false)
    {
        Some(ResponsesDeferredToolsMode::ToolSearch)
    } else {
        None
    };
    let normalize_tool_name = |name: &str| -> String { name.to_owned() };
    let tool_placement =
        split_deferred_tools(context, deferred_tools_mode.is_some(), &normalize_tool_name);
    let deferred_tools: BTreeMap<String, Tool> = tool_placement.deferred.iter().cloned().collect();
    let tool_options = ConvertResponsesToolsOptions {
        // The wire's `strict: null` rides through to every tool entry.
        strict: Some(None),
        supports_strict_mode: Some(supports_strict_mode),
        supports_openai_grammar_tools: Some(supports_openai_grammar_tools),
        ..ConvertResponsesToolsOptions::default()
    };
    let messages = convert_responses_messages(
        model,
        context,
        &codex_tool_call_providers(),
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            deferred_tools: Some(deferred_tools),
            deferred_tools_mode,
            tool_options: Some(tool_options),
        }),
    )
    .map_err(StreamFailure::plain)?;

    let mut body = Map::new();
    body.insert("model".to_owned(), json!(model.id));
    body.insert("store".to_owned(), json!(false));
    body.insert("stream".to_owned(), json!(true));
    body.insert(
        "instructions".to_owned(),
        json!(
            context
                .system_prompt
                .as_deref()
                .filter(|prompt| !prompt.is_empty())
                .unwrap_or("You are a helpful assistant.")
        ),
    );
    body.insert("input".to_owned(), Value::Array(messages));
    body.insert(
        "text".to_owned(),
        json!({ "verbosity": options
            .text_verbosity
            .as_deref()
            .filter(|verbosity| !verbosity.is_empty())
            .unwrap_or("low") }),
    );
    body.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
    if let Some(session_id) = codex_session_id {
        body.insert("prompt_cache_key".to_owned(), json!(session_id));
    }
    body.insert(
        "tool_choice".to_owned(),
        json!(options.tool_choice.map_or("auto", CodexToolChoice::as_str)),
    );
    body.insert("parallel_tool_calls".to_owned(), json!(true));

    if let Some(temperature) = options.temperature {
        body.insert("temperature".to_owned(), json!(temperature));
    }

    if let Some(service_tier) = &options.service_tier {
        body.insert("service_tier".to_owned(), json!(service_tier));
    }

    if !tool_placement.immediate.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(
                convert_responses_tools(&tool_placement.immediate, Some(&tool_options))
                    .map_err(StreamFailure::plain)?,
            ),
        );
    }

    if let Some(reasoning) = build_reasoning_field(model, options) {
        body.insert("reasoning".to_owned(), reasoning);
    }

    Ok(body)
}

/// The `reasoning` request field, upstream's `buildRequestBody` reasoning
/// branch: an explicit effort maps through `thinkingLevelMap` (the off state
/// is the wire's `"none"` effort, which reads `thinkingLevelMap.off`) and
/// carries the summary, a null mapping drops the field, and no explicit
/// effort falls back to the model's off level when it is not null — sending
/// `{"effort": ...}` alone, the object literal upstream's request-less
/// `else if` branch assigns.
fn build_reasoning_field(model: &Model, options: &OpenAiCodexResponsesOptions) -> Option<Value> {
    let level_map = model.thinking_level_map.as_ref();
    let effort = match options.reasoning_effort {
        Some(ModelThinkingLevel::Off) => {
            match level_map.and_then(|map| map.get(&ModelThinkingLevel::Off)) {
                None => Some(json!("none")),
                Some(None) => None,
                Some(Some(mapped)) => Some(json!(mapped.clone())),
            }
        }
        Some(level) => pi_thinking_level(level).and_then(|level| mapped_effort(level_map, level)),
        None => {
            // `model.reasoning && model.thinkingLevelMap?.off !== null`: the
            // mapped-off entry, `"none"` when the map does not name one.
            if !model.reasoning
                || level_map.and_then(|map| map.get(&ModelThinkingLevel::Off)) == Some(&None)
            {
                return None;
            }
            let mapped = level_map
                .and_then(|map| map.get(&ModelThinkingLevel::Off))
                .and_then(|entry| entry.as_deref());
            Some(mapped.map_or_else(|| json!("none"), |mapped| json!(mapped)))
        }
    };
    match options.reasoning_effort {
        Some(_) => effort.map(|effort| {
            json!({
                "effort": effort,
                "summary": options
                    .reasoning_summary
                    .as_deref()
                    .filter(|summary| !summary.is_empty())
                    .unwrap_or("auto"),
            })
        }),
        None => effort.map(|effort| json!({ "effort": effort })),
    }
}

/// The service-tier resolver hook a codex stream passes into the shared
/// processor, upstream's `resolveCodexServiceTier`: a response tier of
/// `"default"` under a client-requested flex or priority keeps the client's
/// tier for pricing, else the response tier falls back to the request's.
fn resolve_codex_service_tier(
    response_tier: Option<&str>,
    request_tier: Option<&str>,
) -> Option<String> {
    if response_tier == Some("default") && matches!(request_tier, Some("flex" | "priority")) {
        return request_tier.map(str::to_owned);
    }
    response_tier
        .map(str::to_owned)
        .or_else(|| request_tier.map(str::to_owned))
}

// ---------------------------------------------------------------------------
// Codex event mapping
// ---------------------------------------------------------------------------

/// One source event through the codex mapper, upstream's `mapCodexEvents`
/// loop body.
enum MappedEvent {
    /// An event without a usable `type`, upstream's `continue`.
    Skipped,
    /// A mapped event the stream forwards.
    Event(Map<String, Value>),
    /// The terminal event, rewritten to `response.completed`; the stream ends
    /// after forwarding it.
    Terminal(Map<String, Value>),
}

/// The `code`/`message` pair a wire error event carries, upstream's
/// `extractCodexEventError`: the top-level fields first, then the nested
/// `error` object's.
fn extract_codex_event_error(event: &Value) -> (Option<String>, Option<String>) {
    let nested = event.get("error").filter(|error| error.is_object());
    let code = event
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| {
            nested
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str)
        })
        .map(str::to_owned);
    let message = event
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| {
            nested
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
        })
        .map(str::to_owned);
    (code, message)
}

/// Apply the codex event mapping to one raw event, upstream's `mapCodexEvents`
/// loop body: wire `error` and `response.failed` events raise the API failure
/// (their `code` rides alongside for the retry classification), the terminal
/// family normalizes the response status and rewrites to
/// `response.completed`, and every other event rides through verbatim.
///
/// # Errors
/// The [`StreamFailure`] a wire error event or `response.failed` terminal
/// carries.
fn map_codex_event(event: &Value, shared: &StreamShared) -> Result<MappedEvent, StreamFailure> {
    let Some(event_type) = event.get("type").and_then(Value::as_str) else {
        return Ok(MappedEvent::Skipped);
    };

    if event_type == "error" {
        let (code, message) = extract_codex_event_error(event);
        let detail = message
            .as_deref()
            .filter(|message| !message.is_empty())
            .or_else(|| code.as_deref().filter(|code| !code.is_empty()))
            .map_or_else(|| event.to_string(), str::to_owned);
        return Err(StreamFailure::api(format!("Codex error: {detail}"), code));
    }

    if event_type == "response.failed" {
        let error = event
            .get("response")
            .and_then(|response| response.get("error"));
        let message = error
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .filter(|message| !message.is_empty())
            .map_or_else(|| "Codex response failed".to_owned(), str::to_owned);
        let code = error
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        return Err(StreamFailure::api(message, code));
    }

    if matches!(
        event_type,
        "response.done" | "response.completed" | "response.incomplete"
    ) {
        if let Some(response) = event
            .get("response")
            .filter(|response| response.is_object())
            && let Some(end_turn) = response.get("end_turn").and_then(Value::as_bool)
        {
            *shared
                .end_turn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(end_turn);
        }
        let mut terminal = event.as_object().cloned().unwrap_or_default();
        // The spread keeps an existing key's position; `type` and `response`
        // are replaced where they already sit.
        terminal.insert("type".to_owned(), json!("response.completed"));
        match terminal.get("response") {
            Some(Value::Object(_)) => {
                let mut response = terminal.get("response").cloned().unwrap_or_default();
                if let Some(object) = response.as_object_mut() {
                    match object
                        .get("status")
                        .and_then(Value::as_str)
                        .and_then(codex_response_status)
                    {
                        Some(status) => {
                            object.insert("status".to_owned(), json!(status));
                        }
                        None => {
                            // An unknown status normalizes to undefined, which
                            // the wire serialization drops.
                            object.remove("status");
                        }
                    }
                    terminal.insert("response".to_owned(), response);
                }
            }
            // A `null` response rides; any other shape never crosses the wire.
            Some(Value::Null) => {}
            _ => {
                terminal.remove("response");
            }
        }
        return Ok(MappedEvent::Terminal(terminal));
    }

    event
        .as_object()
        .cloned()
        .map_or(Ok(MappedEvent::Skipped), |object| {
            Ok(MappedEvent::Event(object))
        })
}

// ---------------------------------------------------------------------------
// WebSocket session cache
// ---------------------------------------------------------------------------

/// The replay state a cached WebSocket carries between turns, upstream's
/// `CachedWebSocketContinuationState`: the last full request body, the
/// response id the connection scoped, and the response's input items.
#[expect(
    clippy::struct_field_names,
    reason = "the `last_*` fields carry upstream's `lastRequestBody`/`lastResponseId`/`lastResponseItems` names verbatim; the spelling is the wire vocabulary"
)]
struct CachedWebSocketContinuationState {
    /// The last request body sent on the connection.
    last_request_body: Map<String, Value>,
    /// The connection-scoped response id continuation rides on.
    last_response_id: String,
    /// The response's converted input items, the baseline the delta prefixes.
    last_response_items: Vec<Value>,
}

/// A cached WebSocket connection, upstream's `CachedWebSocketConnection`.
struct CachedWebSocketConnection {
    /// The parked connection while no stream holds it.
    connection: Option<Box<dyn WebSocketConnection>>,
    /// Whether a stream currently holds the connection.
    busy: bool,
    /// Unix milliseconds when the connection was created, the age-limit
    /// anchor.
    created_at: i64,
    /// The idle-expiry task, cancelled on reuse.
    idle_abort: Option<tokio::task::AbortHandle>,
    /// The continuation state the `websocket-cached` mode sends deltas from.
    continuation: Option<CachedWebSocketContinuationState>,
}

/// The per-session transport bookkeeping, upstream's
/// `OpenAICodexWebSocketDebugStats`. Every number here is transport
/// instrumentation, not wire data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpenAiCodexWebSocketDebugStats {
    /// Stream requests served, upstream's `requests`.
    pub requests: u64,
    /// WebSocket connections opened, upstream's `connectionsCreated`.
    pub connections_created: u64,
    /// WebSocket connections reused from the cache, upstream's
    /// `connectionsReused`.
    pub connections_reused: u64,
    /// Requests that consulted a cached connection's continuation, upstream's
    /// `cachedContextRequests`.
    pub cached_context_requests: u64,
    /// Requests that sent `store: true`, upstream's `storeTrueRequests`;
    /// always zero on this wire, which rejects it.
    pub store_true_requests: u64,
    /// Requests that sent the full input, upstream's `fullContextRequests`.
    pub full_context_requests: u64,
    /// Requests that sent only the input delta, upstream's `deltaRequests`.
    pub delta_requests: u64,
    /// Input items of the last request, upstream's `lastInputItems`.
    pub last_input_items: u64,
    /// Input items of the last delta request, upstream's `lastDeltaInputItems`.
    pub last_delta_input_items: Option<u64>,
    /// Response id of the last delta request, upstream's
    /// `lastPreviousResponseId`.
    pub last_previous_response_id: Option<String>,
    /// Failed WebSocket attempts, upstream's `websocketFailures`.
    pub websocket_failures: u64,
    /// SSE fallbacks taken, upstream's `sseFallbacks`.
    pub sse_fallbacks: u64,
    /// Whether the session currently rides the SSE fallback, upstream's
    /// `websocketFallbackActive`.
    pub websocket_fallback_active: Option<bool>,
    /// The last WebSocket failure's display form, upstream's
    /// `lastWebSocketError`.
    pub last_web_socket_error: Option<String>,
}

/// The process-global Codex WebSocket state: the session connections, the
/// debug stats, and the sessions riding the SSE fallback, upstream's
/// module-level maps and set.
#[derive(Default)]
struct WebSocketState {
    connections: HashMap<String, HashMap<String, CachedWebSocketConnection>>,
    stats: HashMap<String, OpenAiCodexWebSocketDebugStats>,
    sse_fallback_sessions: HashSet<String>,
}

fn lock_state() -> std::sync::MutexGuard<'static, WebSocketState> {
    static STATE: LazyLock<Mutex<WebSocketState>> = LazyLock::new(|| {
        // The guard has no drop-time unregister; the registration persists
        // for the process like upstream's module-level set.
        let _cleanup =
            crate::session_resources::register_session_resource_cleanup(Box::new(|session_id| {
                close_openai_codex_websocket_sessions(session_id);
                Ok(())
            }));
        Mutex::new(WebSocketState::default())
    });
    STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The WebSocket transport a stream connects through: the override a test
/// installs, else the process default.
fn websocket_transport() -> Arc<dyn WebSocketTransport> {
    websocket_transport_override()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .unwrap_or_else(crate::http::default_websocket_transport)
}

fn websocket_transport_override() -> &'static Mutex<Option<Arc<dyn WebSocketTransport>>> {
    static OVERRIDE: LazyLock<Mutex<Option<Arc<dyn WebSocketTransport>>>> =
        LazyLock::new(|| Mutex::new(None));
    &OVERRIDE
}

/// Install the WebSocket transport streams connect through, the port of
/// stubbing `globalThis.WebSocket` upstream; `None` restores the process
/// default.
pub fn set_websocket_transport(transport: Option<Arc<dyn WebSocketTransport>>) {
    *websocket_transport_override()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = transport;
}

/// Close every cached WebSocket connection of a session — or of all sessions
/// when none is named — and drop the session's cache entries, upstream's
/// `closeOpenAICodexWebSocketSessions`.
///
/// Registered as a session resource so session teardown closes the sockets.
pub fn close_openai_codex_websocket_sessions(session_id: Option<&str>) {
    let mut state = lock_state();
    let mut closing: Vec<Box<dyn WebSocketConnection>> = Vec::new();
    let take_entry = |entry: &mut CachedWebSocketConnection| {
        if let Some(handle) = entry.idle_abort.take() {
            handle.abort();
        }
        entry.connection.take()
    };
    match session_id {
        Some(session_id) => {
            if let Some(entries) = state.connections.remove(session_id) {
                for (_, mut entry) in entries {
                    if let Some(connection) = take_entry(&mut entry) {
                        closing.push(connection);
                    }
                }
            }
        }
        None => {
            for (_, entries) in state.connections.drain() {
                for (_, mut entry) in entries {
                    if let Some(connection) = take_entry(&mut entry) {
                        closing.push(connection);
                    }
                }
            }
        }
    }
    drop(state);
    for connection in closing {
        close_websocket_silently(connection, 1000, "debug_close");
    }
}

/// Drop the transport bookkeeping of a session (or all of them), upstream's
/// `resetOpenAICodexWebSocketDebugStats`.
pub fn reset_openai_codex_websocket_debug_stats(session_id: Option<&str>) {
    let mut state = lock_state();
    if let Some(session_id) = session_id {
        state.stats.remove(session_id);
        state.sse_fallback_sessions.remove(session_id);
    } else {
        state.stats.clear();
        state.sse_fallback_sessions.clear();
    }
}

/// The transport bookkeeping a session accumulated, upstream's
/// `getOpenAICodexWebSocketDebugStats`; `None` before the session's first
/// counted request or fallback.
#[must_use]
pub fn get_openai_codex_websocket_debug_stats(
    session_id: &str,
) -> Option<OpenAiCodexWebSocketDebugStats> {
    lock_state().stats.get(session_id).cloned()
}

/// Whether a session rides the SSE fallback, upstream's
/// `isWebSocketSseFallbackActive`.
fn is_websocket_sse_fallback_active(session_id: Option<&str>) -> bool {
    session_id.is_some_and(|session_id| lock_state().sse_fallback_sessions.contains(session_id))
}

/// Count a session's ride on the SSE fallback, upstream's
/// `recordWebSocketSseFallback`. Sessions without one are skipped, the JS
/// falsy guard.
fn record_websocket_sse_fallback(session_id: Option<&str>) {
    let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty()) else {
        return;
    };
    {
        let mut state = lock_state();
        let fallback_active = state.sse_fallback_sessions.contains(session_id);
        let session_stats = get_or_create_stats(&mut state, session_id);
        session_stats.sse_fallbacks += 1;
        session_stats.websocket_fallback_active = Some(fallback_active);
        drop(state);
    }
}

/// Mark a session as riding the SSE fallback and record the WebSocket
/// failure, upstream's `recordWebSocketFailure`.
fn record_websocket_failure(session_id: Option<&str>, error: &StreamFailure) {
    let Some(session_id) = session_id.filter(|session_id| !session_id.is_empty()) else {
        return;
    };
    {
        let mut state = lock_state();
        state.sse_fallback_sessions.insert(session_id.to_owned());
        let session_stats = get_or_create_stats(&mut state, session_id);
        session_stats.websocket_failures += 1;
        session_stats.last_web_socket_error = Some(format_thrown_value(error));
        session_stats.websocket_fallback_active = Some(true);
        drop(state);
    }
}

fn get_or_create_stats<'a>(
    state: &'a mut WebSocketState,
    session_id: &str,
) -> &'a mut OpenAiCodexWebSocketDebugStats {
    state.stats.entry(session_id.to_owned()).or_default()
}

/// Close a WebSocket without observing the close result, upstream's
/// `closeWebSocketSilently`; without a runtime the drop closes the TCP side
/// unframed.
fn close_websocket_silently(mut connection: Box<dyn WebSocketConnection>, code: u16, reason: &str) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let reason = reason.to_owned();
            handle.spawn(async move {
                let _ = connection.close(code, reason).await;
            });
        }
        Err(_) => drop(connection),
    }
}

/// Open one WebSocket connection through the transport, upstream's
/// `connectWebSocket`. The handshake headers ride lowercased the way the
/// `Headers`-to-record round-trip spells them.
///
/// # Errors
/// The handshake failure: an abort, the connect timeout, or a transport
/// failure, each surfaced with the message the wire reports.
async fn connect_websocket(
    url: &str,
    headers: Vec<(String, String)>,
    signal: &CancellationToken,
    connect_timeout_ms: Option<u64>,
) -> Result<Box<dyn WebSocketConnection>, StreamFailure> {
    let record = websocket_handshake_record(headers)?;
    let request = WebSocketRequest {
        url: url.to_owned(),
        headers: record,
        // `connectTimeoutMs > 0` upstream: zero disables the handshake timer,
        // which the seam cannot express — the zero rides the transport default.
        connect_timeout_ms: connect_timeout_ms.filter(|timeout_ms| *timeout_ms > 0),
        signal: signal.clone(),
    };
    websocket_transport()
        .connect(request)
        .await
        .map_err(|error| match error {
            WebSocketError::Aborted => StreamFailure::plain("Request was aborted"),
            WebSocketError::Timeout(ms) => {
                StreamFailure::plain(format!("WebSocket connect timeout after {ms}ms"))
            }
            other => StreamFailure::plain(other.to_string()),
        })
}

/// The close failure message a peer close carries, upstream's
/// `extractWebSocketCloseError`: `WebSocket closed {code} {reason}`, with the
/// too-big close code's default reason spelled out.
fn websocket_close_message(code: u16, reason: &str) -> String {
    let mut message = format!("WebSocket closed {code}");
    if !reason.is_empty() {
        message.push(' ');
        message.push_str(reason);
    } else if code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE {
        message.push_str(" message too big");
    }
    message.trim().to_owned()
}

// ---------------------------------------------------------------------------
// Cached WebSocket acquisition
// ---------------------------------------------------------------------------

/// Take a connection for a stream, upstream's `acquireWebSocket`: cached
/// connections are keyed by session and account, replaced past the backend
/// connection age limit, and checked out per stream; a busy entry or a
/// missing session connects fresh and releases without caching. Returns the
/// checked-out connection, the cache slot it belongs to (`None` for one-shot
/// sockets), and whether the connection was reused.
///
/// # Errors
/// The connect failure; the session-cache paths reconnect through it.
async fn acquire_websocket(
    url: &str,
    headers: Vec<(String, String)>,
    session_id: Option<&str>,
    account_id: &str,
    signal: &CancellationToken,
    connect_timeout_ms: Option<u64>,
) -> Result<
    (
        Option<Box<dyn WebSocketConnection>>,
        Option<(String, String)>,
        bool,
    ),
    StreamFailure,
> {
    // `if (!sessionId)`: a missing or empty session never caches.
    let Some(session) = session_id.filter(|session| !session.is_empty()) else {
        let connection = connect_websocket(url, headers, signal, connect_timeout_ms).await?;
        return Ok((Some(connection), None, false));
    };

    // The idle timer of a cached entry is cleared on every acquire, upstream's
    // `clearTimeout(cached.idleTimer)`.
    let reusable: Option<(Box<dyn WebSocketConnection>, (String, String))> = {
        let mut state = lock_state();
        if let Some(entry) = find_entry_mut(&mut state, session, account_id)
            && let Some(handle) = entry.idle_abort.take()
        {
            handle.abort();
        }
        let mut reusable = None;
        if let Some(entries) = state.connections.get_mut(session)
            && let Some(entry) = entries.get_mut(account_id)
        {
            if !entry.busy && is_websocket_session_expired(entry) {
                let expired = entry.connection.take();
                entries.remove(account_id);
                if entries.is_empty() {
                    state.connections.remove(session);
                }
                if let Some(connection) = expired {
                    close_websocket_silently(connection, 1000, "connection_age_limit");
                }
            } else if !entry.busy && entry.connection.is_some() {
                // The seam carries no readyState: a cached connection in the
                // map is reusable, and a peer-dropped one surfaces on use as a
                // transport failure the fallback machine absorbs.
                entry.busy = true;
                reusable = entry
                    .connection
                    .take()
                    .map(|connection| (connection, (session.to_owned(), account_id.to_owned())));
            }
        }
        drop(state);
        reusable
    };
    if let Some((connection, slot)) = reusable {
        return Ok((Some(connection), Some(slot), true));
    }

    let connection = connect_websocket(url, headers, signal, connect_timeout_ms).await?;
    {
        let mut state = lock_state();
        state
            .connections
            .entry(session.to_owned())
            .or_default()
            .insert(
                account_id.to_owned(),
                CachedWebSocketConnection {
                    connection: None,
                    busy: true,
                    created_at: now_ms(),
                    idle_abort: None,
                    continuation: None,
                },
            );
    }
    Ok((
        Some(connection),
        Some((session.to_owned(), account_id.to_owned())),
        false,
    ))
}

fn find_entry_mut<'a>(
    state: &'a mut WebSocketState,
    session: &str,
    account: &str,
) -> Option<&'a mut CachedWebSocketConnection> {
    state
        .connections
        .get_mut(session)
        .and_then(|entries| entries.get_mut(account))
}

/// Whether a cached connection outlived the backend connection age limit,
/// upstream's `isWebSocketSessionExpired`.
fn is_websocket_session_expired(entry: &CachedWebSocketConnection) -> bool {
    now_ms() - entry.created_at >= SESSION_WEBSOCKET_MAX_AGE_MS
}

/// Schedule the idle-expiry close of a released connection, upstream's
/// `scheduleSessionWebSocketExpiry`: five minutes of idleness closes the
/// socket and drops the cache entry, unless a stream re-acquired it first.
fn schedule_session_websocket_expiry(session: &str, account: &str) {
    let session_owned = session.to_owned();
    let account_owned = account.to_owned();
    let handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(SESSION_WEBSOCKET_CACHE_TTL_MS)).await;
        let connection = {
            let mut state = lock_state();
            let Some(entries) = state.connections.get_mut(&session_owned) else {
                return;
            };
            let Some(entry) = entries.get_mut(&account_owned) else {
                return;
            };
            if entry.busy {
                return;
            }
            let connection = entry.connection.take();
            entries.remove(&account_owned);
            if entries.is_empty() {
                state.connections.remove(&session_owned);
            }
            connection
        };
        if let Some(mut connection) = connection {
            let _ = connection.close(1000, "idle_timeout".to_owned()).await;
        }
    });
    let mut state = lock_state();
    if let Some(entry) = find_entry_mut(&mut state, session, account) {
        entry.idle_abort = Some(handle.abort_handle());
    } else {
        handle.abort();
    }
}

/// Return an acquired connection, upstream's `release`: kept connections go
/// back into the session cache with a fresh idle timer, dropped ones close.
fn release_websocket(
    slot: Option<(String, String)>,
    connection: Option<Box<dyn WebSocketConnection>>,
    keep: bool,
) {
    let Some(connection) = connection else {
        // The pump dropped the connection mid-stream; drop the entry so the
        // next acquire reconnects.
        if let Some((session, account)) = slot {
            let mut state = lock_state();
            if let Some(entries) = state.connections.get_mut(&session) {
                entries.remove(&account);
                if entries.is_empty() {
                    state.connections.remove(&session);
                }
            }
        }
        return;
    };
    let Some((session, account)) = slot else {
        close_websocket_silently(connection, 1000, "done");
        return;
    };
    if !keep {
        close_websocket_silently(connection, 1000, "done");
        let mut state = lock_state();
        if let Some(entries) = state.connections.get_mut(&session) {
            entries.remove(&account);
            if entries.is_empty() {
                state.connections.remove(&session);
            }
        }
        drop(state);
        return;
    }
    {
        let mut state = lock_state();
        if let Some(entry) = find_entry_mut(&mut state, &session, &account) {
            entry.connection = Some(connection);
            entry.busy = false;
        }
    }
    schedule_session_websocket_expiry(&session, &account);
}

// ---------------------------------------------------------------------------
// Cached-context continuation
// ---------------------------------------------------------------------------

/// The request body without its `input` and `previous_response_id` fields,
/// upstream's `requestBodyWithoutInput`.
fn request_body_without_input(body: &Map<String, Value>) -> Map<String, Value> {
    body.iter()
        .filter(|(key, _)| key.as_str() != "input" && key.as_str() != "previous_response_id")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Whether two response-input lists serialize the same, upstream's
/// `responseInputsEqual`.
fn response_inputs_equal(a: Option<&Value>, b: Option<&Value>) -> bool {
    let empty = Value::Array(Vec::new());
    let a = a.unwrap_or(&empty);
    let b = b.unwrap_or(&empty);
    serde_json::to_string(a).unwrap_or_default() == serde_json::to_string(b).unwrap_or_default()
}

/// Whether two request bodies agree outside their input fields, upstream's
/// `requestBodiesMatchExceptInput`.
fn request_bodies_match_except_input(a: &Map<String, Value>, b: &Map<String, Value>) -> bool {
    serde_json::to_string(&Value::Object(request_body_without_input(a))).unwrap_or_default()
        == serde_json::to_string(&Value::Object(request_body_without_input(b))).unwrap_or_default()
}

/// The input delta a cached continuation sends, upstream's
/// `getCachedWebSocketInputDelta`: the current input's tail beyond the
/// baseline the connection already holds, when the rest of the body matches.
fn get_cached_websocket_input_delta(
    body: &Map<String, Value>,
    continuation: &CachedWebSocketContinuationState,
) -> Option<Value> {
    if !request_bodies_match_except_input(body, &continuation.last_request_body) {
        return None;
    }

    let current_input = body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut baseline: Vec<Value> = continuation
        .last_request_body
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    baseline.extend(continuation.last_response_items.iter().cloned());
    let baseline_len = baseline.len();
    if current_input.len() < baseline_len {
        return None;
    }

    let prefix = Value::Array(current_input[..baseline_len].to_vec());
    if !response_inputs_equal(Some(&prefix), Some(&Value::Array(baseline))) {
        return None;
    }

    Some(Value::Array(current_input[baseline_len..].to_vec()))
}

/// Rewrite the request body around a cached connection's continuation,
/// upstream's `buildCachedWebSocketRequestBody`: matching bodies send only
/// the input delta with the connection-scoped `previous_response_id`, and a
/// non-matching body drops the continuation and sends in full.
fn build_cached_websocket_request_body(
    entry: &mut CachedWebSocketConnection,
    body: &Map<String, Value>,
) -> Map<String, Value> {
    let Some(continuation) = entry.continuation.take() else {
        return body.clone();
    };

    let delta = get_cached_websocket_input_delta(body, &continuation);
    if let (Some(delta), false) = (delta, continuation.last_response_id.is_empty()) {
        let mut request_body = body.clone();
        request_body.insert(
            "previous_response_id".to_owned(),
            json!(continuation.last_response_id),
        );
        request_body.insert("input".to_owned(), delta);
        request_body
    } else {
        entry.continuation = None;
        body.clone()
    }
}

/// The input items a completed response replays through the cached
/// connection, upstream's `processWebSocketStream` response-item conversion:
/// the assistant message's items minus the tool-output entries the tool
/// results carry.
///
/// # Errors
/// The message-conversion rejection the replayed blocks hit.
fn build_websocket_continuation_items(
    model: &Model,
    output: &AssistantMessage,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Vec<Value>, StreamFailure> {
    let items = convert_responses_messages(
        model,
        &Context {
            messages: vec![Message::Assistant(output.clone())],
            ..Context::default()
        },
        &codex_tool_call_providers(),
        Some(&ConvertResponsesMessagesOptions {
            include_system_prompt: Some(false),
            grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
            ..ConvertResponsesMessagesOptions::default()
        }),
    )
    .map_err(StreamFailure::plain)?;
    Ok(items
        .into_iter()
        .filter(|item| {
            !matches!(
                item.get("type").and_then(Value::as_str),
                Some("function_call_output" | "custom_tool_call_output")
            )
        })
        .collect())
}

/// Clear a cached connection's continuation after a failed attempt, upstream's
/// catch-block `entry.continuation = undefined`.
fn clear_cached_websocket_continuation(session: &str, account: &str) {
    let mut state = lock_state();
    if let Some(entry) = find_entry_mut(&mut state, session, account) {
        entry.continuation = None;
    }
}

// ---------------------------------------------------------------------------
// Mapped event pump
// ---------------------------------------------------------------------------

/// The state the mapped event pump shares with the stream runner, upstream's
/// closure-captured `startEmitted`/`websocketStarted` and the mapper's
/// `output.endTurn` write, plus the mapped stream's failure slot.
#[derive(Default)]
struct StreamShared {
    /// Whether a `start` event rode out already, upstream's `startEmitted`.
    start_emitted: AtomicBool,
    /// Whether the WebSocket attempt pushed output events, upstream's
    /// `websocketStarted`.
    websocket_started: AtomicBool,
    /// The terminal response's `end_turn` when it carried one.
    end_turn: Mutex<Option<bool>>,
    /// The mapped stream's failure, read after the shared processor returns.
    failure: Mutex<Option<StreamFailure>>,
}

impl StreamShared {
    fn store_failure(&self, failure: StreamFailure) {
        *self
            .failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(failure);
    }

    fn take_failure(&self) -> Option<StreamFailure> {
        std::mem::take(
            &mut *self
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    fn take_end_turn(&self) -> Option<bool> {
        std::mem::take(
            &mut *self
                .end_turn
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

/// Push the `start` event once, upstream's `if (!startEmitted) stream.push({
/// type: "start", partial: output })` guard.
fn push_start_once(
    shared: &StreamShared,
    forward: &AssistantMessageEventStream,
    partial: &AssistantMessage,
) {
    if !shared.start_emitted.swap(true, Ordering::AcqRel) {
        forward.push(AssistantMessageEvent::Start {
            partial: partial.clone(),
        });
    }
}

/// The checked-out socket slot a mapped pump returns its WebSocket
/// connection to when the stream ends.
type SocketSlot = Arc<Mutex<Option<Box<dyn WebSocketConnection>>>>;

/// One pump step's outcome.
enum StepOutcome {
    /// A mapped event to re-frame.
    Frame(Map<String, Value>),
    /// The terminal event; the stream ends after forwarding it.
    Terminal(Map<String, Value>),
    /// An event without a usable type; the pump reads on.
    Skip,
    /// A failure was stored; the pump ends.
    Failed,
    /// The source ended cleanly without a terminal event; the shared
    /// processor reports the missing terminal itself.
    CleanEnd,
}

/// One pump poll's result.
enum PumpStep {
    /// A mapped event re-framed as a `data:` line.
    Frame(Map<String, Value>),
    /// The stream ended.
    End,
}

/// The WebSocket side of a mapped pump: the checked-out connection, the slot
/// it returns to, and the read state. The connection races the request's
/// cancellation token itself, surfacing aborts as [`WebSocketError::Aborted`].
struct WebSocketSide {
    /// The slot the connection returns to when the pump ends or drops.
    slot: Arc<Mutex<Option<Box<dyn WebSocketConnection>>>>,
    /// The connection the pump reads while it lives.
    connection: Option<Box<dyn WebSocketConnection>>,
    /// The read idleness cap, upstream's `idleTimeoutMs`; zero reads unbounded.
    idle_timeout_ms: Option<u64>,
    /// Whether a terminal-family message arrived, upstream's `sawCompletion`.
    saw_completion: bool,
}

/// The mapped pump of one codex stream, upstream's
/// `mapCodexEvents(parseSSE(...)/parseWebSocket(...))` feeding
/// `processResponsesStream`: it reads its transport's raw events, applies the
/// codex event mapping, and re-frames the mapped events as `data:` lines of
/// the synthetic SSE body the shared Responses processor consumes.
struct MappedPump {
    source: PumpSource,
    shared: Arc<StreamShared>,
    forward: AssistantMessageEventStream,
    start_partial: AssistantMessage,
    finished: bool,
}

enum PumpSource {
    Sse(SseStream),
    WebSocket(WebSocketSide),
}

impl Drop for MappedPump {
    fn drop(&mut self) {
        if let PumpSource::WebSocket(side) = &mut self.source
            && let Some(connection) = side.connection.take()
        {
            *side
                .slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(connection);
        }
    }
}

impl MappedPump {
    async fn step(&mut self) -> PumpStep {
        loop {
            if self.finished {
                return PumpStep::End;
            }
            let websocket = matches!(self.source, PumpSource::WebSocket(_));
            let outcome = match &mut self.source {
                PumpSource::Sse(source) => step_sse_source(source, &self.shared).await,
                PumpSource::WebSocket(side) => step_websocket_source(side, &self.shared).await,
            };
            match outcome {
                StepOutcome::Skip => {}
                StepOutcome::Failed => return PumpStep::End,
                StepOutcome::CleanEnd => {
                    self.finished = true;
                    return PumpStep::End;
                }
                StepOutcome::Frame(event) => {
                    self.mark_started(websocket);
                    return PumpStep::Frame(event);
                }
                StepOutcome::Terminal(event) => {
                    self.finished = true;
                    self.mark_started(websocket);
                    return PumpStep::Frame(event);
                }
            }
        }
    }

    /// The first mapped event's `start` ride, upstream's
    /// `startWebSocketOutputOnFirstEvent` + `onStart`: the start event pushes
    /// once per stream, and the WebSocket attempt marks itself as having
    /// started so the fallback machine knows events rode out.
    fn mark_started(&self, websocket: bool) {
        push_start_once(&self.shared, &self.forward, &self.start_partial);
        if websocket {
            self.shared.websocket_started.store(true, Ordering::Release);
        }
    }
}

/// One SSE frame through the mapper, upstream's `parseSSE` loop body.
async fn step_sse_source(source: &mut SseStream, shared: &StreamShared) -> StepOutcome {
    loop {
        match source.next().await {
            Ok(Some(frame)) => {
                let data = frame.data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(data) {
                    Ok(event) => event,
                    Err(error) => {
                        shared.store_failure(StreamFailure::protocol(format!(
                            "Invalid Codex SSE JSON: {error}"
                        )));
                        return StepOutcome::Failed;
                    }
                };
                match map_codex_event(&event, shared) {
                    Ok(MappedEvent::Skipped) => {}
                    Ok(MappedEvent::Event(event)) => return StepOutcome::Frame(event),
                    Ok(MappedEvent::Terminal(event)) => return StepOutcome::Terminal(event),
                    Err(failure) => {
                        shared.store_failure(failure);
                        return StepOutcome::Failed;
                    }
                }
            }
            // EOF without a terminal event: the shared processor reports the
            // missing terminal itself, like upstream's `processResponsesStream`
            // check does.
            Ok(None) => return StepOutcome::CleanEnd,
            Err(HttpError::Aborted) => {
                shared.store_failure(StreamFailure::plain("Request was aborted"));
                return StepOutcome::Failed;
            }
            Err(error) => {
                shared.store_failure(StreamFailure::plain(error.to_string()));
                return StepOutcome::Failed;
            }
        }
    }
}

/// One WebSocket read through the mapper, upstream's `parseWebSocket` loop
/// body: the idle timeout races each read, a completion message marks the
/// stream saw its terminal, and close/error events become the failures the
/// fallback machine classifies.
async fn step_websocket_source(side: &mut WebSocketSide, shared: &StreamShared) -> StepOutcome {
    let Some(mut connection) = side.connection.take() else {
        shared.store_failure(StreamFailure::plain(
            "WebSocket connection is not available",
        ));
        return StepOutcome::Failed;
    };
    let read = connection.next_message();
    let read_outcome = match side.idle_timeout_ms {
        Some(idle) if idle > 0 => tokio::select! {
            message = read => WebSocketRead::Message(message),
            () = tokio::time::sleep(Duration::from_millis(idle)) => WebSocketRead::Idle,
        },
        _ => WebSocketRead::Message(read.await),
    };
    match read_outcome {
        WebSocketRead::Idle => {
            let idle = side.idle_timeout_ms.unwrap_or_default();
            let _ = connection.close(1000, "idle_timeout".to_owned()).await;
            side.connection = Some(connection);
            shared.store_failure(StreamFailure::plain(format!(
                "WebSocket idle timeout after {idle}ms"
            )));
            StepOutcome::Failed
        }
        WebSocketRead::Message(Ok(WebSocketMessage::Text(text))) => {
            side.connection = Some(connection);
            handle_websocket_text(side, shared, &text)
        }
        WebSocketRead::Message(Ok(WebSocketMessage::Binary(data))) => {
            side.connection = Some(connection);
            let text = String::from_utf8_lossy(&data);
            handle_websocket_text(side, shared, text.as_ref())
        }
        WebSocketRead::Message(Err(WebSocketError::Aborted)) => {
            side.connection = Some(connection);
            shared.store_failure(StreamFailure::plain("Request was aborted"));
            StepOutcome::Failed
        }
        WebSocketRead::Message(Err(WebSocketError::Closed { code, reason, .. })) => {
            side.connection = Some(connection);
            if side.saw_completion {
                // Unreachable in practice: the pump stops reading after the
                // terminal event, and a completion message already marked it.
                StepOutcome::CleanEnd
            } else {
                shared.store_failure(StreamFailure::plain(websocket_close_message(code, &reason)));
                StepOutcome::Failed
            }
        }
        WebSocketRead::Message(Err(other)) => {
            side.connection = Some(connection);
            shared.store_failure(StreamFailure::plain(other.to_string()));
            StepOutcome::Failed
        }
    }
}

/// One WebSocket message's JSON through the mapper, upstream's
/// `parseWebSocket` `onMessage` listener.
fn handle_websocket_text(
    side: &mut WebSocketSide,
    shared: &StreamShared,
    text: &str,
) -> StepOutcome {
    let parsed: Value = match serde_json::from_str(text) {
        Ok(parsed) => parsed,
        Err(error) => {
            shared.store_failure(StreamFailure::protocol(format!(
                "Invalid Codex WebSocket JSON: {error}"
            )));
            return StepOutcome::Failed;
        }
    };
    if matches!(
        parsed.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.done" | "response.incomplete")
    ) {
        side.saw_completion = true;
    }
    match map_codex_event(&parsed, shared) {
        Ok(MappedEvent::Skipped) => StepOutcome::Skip,
        Ok(MappedEvent::Event(event)) => StepOutcome::Frame(event),
        Ok(MappedEvent::Terminal(event)) => StepOutcome::Terminal(event),
        Err(failure) => {
            shared.store_failure(failure);
            StepOutcome::Failed
        }
    }
}

/// What a WebSocket read settled to, the idle race of upstream's
/// `parseWebSocket` wait.
enum WebSocketRead {
    Message(Result<WebSocketMessage, WebSocketError>),
    Idle,
}

/// The synthetic SSE response the shared Responses processor consumes: one
/// `data:` line per mapped event.
fn mapped_stream_response(pump: MappedPump) -> HttpResponse {
    let body = HttpByteStream::new(unfold(pump, |mut pump| async move {
        match pump.step().await {
            PumpStep::Frame(event) => {
                let frame = Bytes::from(format!("data: {}\n\n", Value::Object(event)));
                Some((Ok(frame), pump))
            }
            PumpStep::End => None,
        }
    }));
    HttpResponse {
        status: 200,
        headers: Vec::new(),
        body,
    }
}

/// Process the mapped stream, upstream's
/// `processResponsesStream(mapCodexEvents(...), ...)`: the mapped events ride
/// the synthetic SSE body, and a stored failure replaces the shared
/// processor's missing-terminal error so the wire's error code survives.
async fn run_mapped_stream(
    pump: MappedPump,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    model: &Model,
    stream_options: &OpenAiResponsesStreamOptions,
) -> Result<(), StreamFailure> {
    let shared = Arc::clone(&pump.shared);
    let response = mapped_stream_response(pump);
    let outcome =
        process_responses_stream(response, output, events, model, Some(stream_options)).await;
    match outcome {
        Ok(()) => Ok(()),
        Err(message) => shared
            .take_failure()
            .map_or_else(|| Err(StreamFailure::plain(message)), Err),
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// The OpenAI Codex Responses [`crate::types::ProviderStreams`], upstream's
/// module-level `stream`/`streamSimple` exports behind the uniform dispatch.
#[derive(Debug, Default)]
pub struct OpenAiCodexResponsesStreams;

crate::api::adapter_belt::impl_provider_streams!(
    OpenAiCodexResponsesStreams,
    OpenAiCodexResponsesOptions
);
/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage and `stopReason: "pending"`.
fn initial_output(model: &Model) -> AssistantMessage {
    crate::api::request_seam::initial_output(model, model.api.clone(), None)
}

/// Stream an assistant response over the ChatGPT Codex Responses wire,
/// upstream's `stream`. The stream returns live; request setup, model, and
/// runtime failures arrive as its `error` event.
#[must_use]
pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&OpenAiCodexResponsesOptions>,
) -> AssistantMessageEventStream {
    let options = options.cloned();
    let signal = options.as_ref().map_or_else(
        || TransportOptions::default().signal(),
        |options| options.transport_options.signal(),
    );
    spawned_stream(
        model,
        context,
        signal,
        initial_output(model),
        move |model, context, output, forward| {
            Box::pin(async move {
                let options = options.unwrap_or_default();
                run_stream(&model, &context, &options, output, &forward)
                    .await
                    .map_err(|failure| failure.message)
            })
        },
    )
}

/// Run one stream to completion: resolve the credential and account, build
/// the request, ride the WebSocket transport with its SSE fallback, retry
/// the SSE path on transient failures, and settle the final message.
#[expect(
    clippy::too_many_lines,
    reason = "the transport loop mirrors upstream's stream() body: the WebSocket retries, the fallback record, and the SSE retry loop are one sequence the tests replay"
)]
async fn run_stream(
    model: &Model,
    context: &Context,
    options: &OpenAiCodexResponsesOptions,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
) -> Result<(), StreamFailure> {
    let api_key = options
        .api_key
        .as_deref()
        .filter(|api_key| !api_key.is_empty())
        .ok_or_else(|| {
            StreamFailure::plain(format!("No API key for provider: {}", model.provider.0))
        })?;
    let account_id = extract_account_id(api_key)?;
    let supports_openai_grammar_tools = model
        .compat
        .as_ref()
        .and_then(|compat| compat.supports_openai_grammar_tools)
        .unwrap_or(false);
    let grammar_tool_input_properties = create_grammar_tool_input_properties(
        context.tools.as_deref(),
        supports_openai_grammar_tools,
    );
    let cache_session_id = (options.cache_retention != Some(CacheRetention::None))
        .then(|| options.session_id.clone())
        .flatten();
    let codex_session_id =
        clamp_openai_prompt_cache_key(cache_session_id.as_deref()).map(str::to_owned);
    let mut body = build_request_body(
        model,
        context,
        options,
        codex_session_id.as_deref(),
        &grammar_tool_input_properties,
    )?;
    if let Some(hook) = &options.transport_options.on_payload
        && let Some(Value::Object(next)) =
            hook.call(Value::Object(body.clone()), model.clone()).await
    {
        body = next;
    }
    let websocket_request_id = codex_session_id
        .clone()
        .filter(|session_id| !session_id.is_empty())
        .map_or_else(
            || uuidv7(None).map_err(|error| StreamFailure::plain(error.to_string())),
            Ok,
        )?;
    let sse_headers = build_sse_headers(
        model.headers.as_ref(),
        options.headers.as_ref(),
        &account_id,
        api_key,
        codex_session_id.as_deref(),
    );
    let websocket_headers = build_websocket_headers(
        model.headers.as_ref(),
        options.headers.as_ref(),
        &account_id,
        api_key,
        &websocket_request_id,
    );
    let body_json = Value::Object(body.clone()).to_string();
    let signal = options.transport_options.signal();
    let transport = options.transport.unwrap_or(Transport::Auto);
    let shared = Arc::new(StreamShared::default());

    let websocket_disabled_for_session = transport != Transport::Sse
        && is_websocket_sse_fallback_active(cache_session_id.as_deref());
    if websocket_disabled_for_session {
        record_websocket_sse_fallback(cache_session_id.as_deref());
    }

    // The shared processor options both transports ride, upstream's
    // `{ serviceTier, grammarToolInputProperties, resolveServiceTier,
    // applyServiceTierPricing }` stream options.
    let resolve_service_tier: ResolveServiceTier = Arc::new(resolve_codex_service_tier);
    let stream_options = OpenAiResponsesStreamOptions {
        service_tier: options.service_tier.clone(),
        grammar_tool_input_properties: Some(grammar_tool_input_properties.clone()),
        resolve_service_tier: Some(resolve_service_tier),
        apply_service_tier_pricing: Some(pricing_hook(model.clone())),
    };
    let idle_timeout_ms = options.timeout_ms.filter(|timeout_ms| *timeout_ms > 0);

    if transport != Transport::Sse && !websocket_disabled_for_session {
        let mut retried_websocket_connection_limit = false;
        let mut retried_missing_websocket_continuation = false;
        loop {
            shared.websocket_started.store(false, Ordering::Release);
            match process_websocket_stream(
                &resolve_codex_websocket_url(Some(&model.base_url))?,
                &body,
                &websocket_headers,
                output,
                events,
                model,
                &shared,
                &stream_options,
                &grammar_tool_input_properties,
                options,
                cache_session_id.as_deref(),
                &account_id,
                idle_timeout_ms,
                &signal,
            )
            .await
            {
                Ok(()) => {
                    if signal.is_cancelled() {
                        return Err(StreamFailure::plain("Request was aborted"));
                    }
                    if let Some(end_turn) = shared.take_end_turn() {
                        output.end_turn = Some(end_turn);
                    }
                    assert_successful_output(output)?;
                    events.push(AssistantMessageEvent::Done {
                        reason: output.stop_reason,
                        message: output.clone(),
                    });
                    return Ok(());
                }
                Err(failure) => {
                    let aborted = signal.is_cancelled();
                    let websocket_started = shared.websocket_started.load(Ordering::Acquire);
                    let connection_limit_before_start =
                        !websocket_started && failure.is_websocket_connection_limit();
                    let previous_response_not_found = failure.is_previous_response_not_found();
                    if !aborted
                        && previous_response_not_found
                        && !retried_missing_websocket_continuation
                    {
                        retried_missing_websocket_continuation = true;
                        continue;
                    }
                    if !aborted
                        && connection_limit_before_start
                        && !retried_websocket_connection_limit
                    {
                        retried_websocket_connection_limit = true;
                        continue;
                    }
                    if aborted || (failure.is_non_transport() && !connection_limit_before_start) {
                        return Err(failure);
                    }
                    let mut details = BTreeMap::from([
                        (
                            "configuredTransport".to_owned(),
                            json!(transport_wire(transport)),
                        ),
                        ("eventsEmitted".to_owned(), json!(websocket_started)),
                        (
                            "phase".to_owned(),
                            json!(if websocket_started {
                                "after_message_stream_start"
                            } else {
                                "before_message_stream_start"
                            }),
                        ),
                        ("requestBytes".to_owned(), json!(body_json.len())),
                    ]);
                    if !websocket_started {
                        details.insert("fallbackTransport".to_owned(), json!("sse"));
                    }
                    append_assistant_message_diagnostic(
                        output,
                        create_assistant_message_diagnostic(
                            "provider_transport_failure",
                            &failure,
                            Some(details),
                            now_ms(),
                        ),
                    );
                    record_websocket_failure(cache_session_id.as_deref(), &failure);
                    if websocket_started {
                        return Err(failure);
                    }
                    record_websocket_sse_fallback(cache_session_id.as_deref());
                    break;
                }
            }
        }
    }

    // Compress the request body once for the SSE path. The Codex backend
    // decodes `Content-Encoding: zstd`; the WebSocket transport above sends
    // the uncompressed JSON frame, matching the official Codex client.
    let sse_body = compress_request_body_zstd(&body_json);
    let mut sse_headers = sse_headers;
    push_header(&mut sse_headers, "content-encoding", "zstd");

    // Fetch with retry logic for rate limits and transient errors, upstream's
    // bespoke retry loop.
    let mut response: Option<HttpResponse> = None;
    let max_retries = options.max_retries.unwrap_or(DEFAULT_MAX_RETRIES);
    let http_timeout_ms = options.timeout_ms;
    for attempt in 0..=max_retries {
        if signal.is_cancelled() {
            return Err(StreamFailure::plain("Request was aborted"));
        }
        let request = HttpRequest {
            method: HttpMethod::Post,
            url: resolve_codex_url(Some(&model.base_url)),
            headers: sse_headers.clone(),
            body: Some(sse_body.clone()),
            // The combined abort signal of the TS attempt collapses onto the
            // seam's total request timeout plus the caller token.
            timeout_ms: http_timeout_ms.filter(|timeout_ms| *timeout_ms > 0),
            signal: signal.clone(),
        };
        match sse_attempt(
            model,
            options,
            request,
            http_timeout_ms,
            &signal,
            attempt,
            max_retries,
        )
        .await
        {
            SseAttempt::Response(fresh) => {
                response = Some(fresh);
                break;
            }
            SseAttempt::Retried => {}
            SseAttempt::Abort => return Err(StreamFailure::plain("Request was aborted")),
            SseAttempt::DelayCap(message) => return Err(StreamFailure::plain(message)),
            SseAttempt::Message(message) => {
                // The catch block: every thrown error retries unless the
                // delay cap threw or the message names the usage limit.
                if attempt < max_retries && !message.contains("usage limit") {
                    if sleep(exponential_backoff_ms(attempt), &signal)
                        .await
                        .is_err()
                    {
                        return Err(StreamFailure::plain("Request was aborted"));
                    }
                    continue;
                }
                return Err(StreamFailure::plain(message));
            }
        }
    }

    let Some(response) = response else {
        // The loop exits only with a response or an error; this is the
        // exhausted-retries safety net upstream's `throw lastError ?? ...`
        // covers.
        return Err(StreamFailure::plain("Failed after retries"));
    };

    push_start_once(&shared, events, output);

    let pump = MappedPump {
        source: PumpSource::Sse(SseStream::new(response.body)),
        shared: Arc::clone(&shared),
        forward: events.clone(),
        start_partial: output.clone(),
        finished: false,
    };
    run_mapped_stream(pump, output, events, model, &stream_options).await?;

    if signal.is_cancelled() {
        return Err(StreamFailure::plain("Request was aborted"));
    }
    if let Some(end_turn) = shared.take_end_turn() {
        output.end_turn = Some(end_turn);
    }
    assert_successful_output(output)?;
    events.push(AssistantMessageEvent::Done {
        reason: output.stop_reason,
        message: output.clone(),
    });
    Ok(())
}

/// The settled outcome of one SSE request attempt, upstream's try/catch shape:
/// a response, a retry, or a thrown error the catch block may retry.
enum SseAttempt {
    /// A 2xx response ready to stream.
    Response(HttpResponse),
    /// A retryable failure slept its delay; the loop continues.
    Retried,
    /// The caller aborted before or during the attempt.
    Abort,
    /// A thrown Error the catch block retries unless the message names the
    /// usage limit.
    Message(String),
    /// The server-requested delay exceeded the cap; the request fails
    /// immediately, upstream's `RetryDelayExceededError`.
    DelayCap(String),
}

/// One SSE request attempt, upstream's retry loop's try block: dispatch with
/// the header timeout, fire the response hook, retry retryable failures after
/// the server-requested (or exponential) delay, and raise the friendly error
/// otherwise.
async fn sse_attempt(
    model: &Model,
    options: &OpenAiCodexResponsesOptions,
    request: HttpRequest,
    http_timeout_ms: Option<u64>,
    signal: &CancellationToken,
    attempt: u32,
    max_retries: u32,
) -> SseAttempt {
    let http_client = options.transport_options.client();
    let response = match http_client.execute(request).await {
        Ok(response) => response,
        Err(HttpError::Aborted) => return SseAttempt::Abort,
        Err(HttpError::Timeout) if http_timeout_ms.is_some() && !signal.is_cancelled() => {
            let timeout_ms = http_timeout_ms.unwrap_or_default();
            return SseAttempt::Message(format!(
                "Codex SSE response headers timed out after {timeout_ms}ms"
            ));
        }
        Err(error) => return SseAttempt::Message(error.to_string()),
    };

    fire_response_hook(
        &options.transport_options,
        response.status,
        &response.headers,
        model.clone(),
    )
    .await;

    if (200..300).contains(&response.status) {
        return SseAttempt::Response(response);
    }

    let error_text = read_body_text(response.body).await.unwrap_or_default();
    if attempt < max_retries && is_retryable_error(response.status, &error_text) {
        let delay_ms = match get_retry_after_delay_ms(&response.headers) {
            Some(delay_ms) => match validate_retry_delay_ms(delay_ms, options.max_retry_delay_ms) {
                Ok(delay_ms) => delay_ms,
                Err(failure) => return SseAttempt::DelayCap(failure.message),
            },
            None => exponential_backoff_ms(attempt),
        };
        return match sleep(delay_ms, signal).await {
            Ok(()) => SseAttempt::Retried,
            Err(_) => SseAttempt::Abort,
        };
    }

    let (message, friendly) = parse_error_response(response.status, &error_text);
    SseAttempt::Message(friendly.unwrap_or(message))
}

/// The friendly failure message of a non-2xx response, upstream's
/// `parseErrorResponse`: the ChatGPT usage-limit wording when the error code
/// or the 429 status names a spent quota, else the error body's message, else
/// the raw body.
fn parse_error_response(status: u16, raw: &str) -> (String, Option<String>) {
    let mut message = if raw.is_empty() {
        "Request failed".to_owned()
    } else {
        raw.to_owned()
    };
    let mut friendly_message = None;
    let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
        return (message, None);
    };
    let Some(error) = parsed.get("error").filter(|error| error.is_object()) else {
        return (message, None);
    };
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| error.get("type").and_then(Value::as_str))
        .unwrap_or_default();
    if USAGE_LIMIT_CODE_PATTERN.is_match(code) || status == 429 {
        let plan = error
            .get("plan_type")
            .and_then(Value::as_str)
            .map(|plan| format!(" ({} plan)", plan.to_lowercase()))
            .unwrap_or_default();
        let minutes = error.get("resets_at").and_then(Value::as_f64).map(|resets_at| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "the wall-clock enters the JS number math the friendly message composes"
            )]
            let now_ms_f64 = now_ms() as f64;
            // `resets_at * 1000 - Date.now()`: separate float ops like the JS
            // arithmetic, whose per-op rounding the message wording pins.
            #[expect(
                clippy::suboptimal_flops,
                reason = "the fused multiply-add's intermediate rounding would drift from the JS math the message composes"
            )]
            let minutes = ((resets_at * 1000.0 - now_ms_f64) / 60_000.0).round().max(0.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the minutes are clamped non-negative whole minutes before the cast"
            )]
            let minutes = minutes as u64;
            minutes
        });
        let when = minutes
            .map(|minutes| format!(" Try again in ~{minutes} min."))
            .unwrap_or_default();
        friendly_message = Some(
            format!("You have hit your ChatGPT usage limit{plan}.{when}")
                .trim()
                .to_owned(),
        );
    }
    // `err.message || friendlyMessage || message`: the first non-empty
    // string wins.
    message = error
        .get("message")
        .and_then(Value::as_str)
        .filter(|message| !message.is_empty())
        .map_or_else(
            || friendly_message.clone().unwrap_or(message),
            str::to_owned,
        );
    (message, friendly_message)
}

/// The terminal assertion before a stream settles as `done`, upstream's
/// `assertSuccessfulOutput`.
///
/// # Errors
/// The pending- and failure-stop-reason messages the TS assertion throws.
fn assert_successful_output(output: &AssistantMessage) -> Result<(), StreamFailure> {
    if output.stop_reason == StopReason::Pending {
        return Err(StreamFailure::plain(
            "Codex stream ended without a stop reason",
        ));
    }
    if matches!(output.stop_reason, StopReason::Error | StopReason::Aborted) {
        return Err(StreamFailure::plain(
            output
                .error_message
                .clone()
                .unwrap_or_else(|| "An unknown error occurred".to_owned()),
        ));
    }
    Ok(())
}

/// Ride one WebSocket attempt, upstream's `processWebSocketStream`: acquire
/// (fresh or cached) a connection, send the `response.create` frame with the
/// cached-context delta when one applies, process the mapped events, and
/// store or clear the continuation before releasing the connection. The
/// shared processor options ride in so the service-tier hooks apply.
///
/// # Errors
/// The connect, send, mapped-stream, and continuation-conversion failures.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the parameters mirror upstream's processWebSocketStream signature plus the seam handles the closure captures it held, and the attempt mirrors its body: acquire, cached-context rewrite, stats, send, continuation store, and release are one sequence the tests replay"
)]
async fn process_websocket_stream(
    url: &str,
    full_body: &Map<String, Value>,
    websocket_headers: &[(String, String)],
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    model: &Model,
    shared: &Arc<StreamShared>,
    stream_options: &OpenAiResponsesStreamOptions,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    options: &OpenAiCodexResponsesOptions,
    cache_session_id: Option<&str>,
    account_id: &str,
    idle_timeout_ms: Option<u64>,
    signal: &CancellationToken,
) -> Result<(), StreamFailure> {
    let transport = options.transport.unwrap_or(Transport::Auto);
    let use_cached_context = matches!(transport, Transport::WebsocketCached | Transport::Auto);
    let (connection, slot, reused) = acquire_websocket(
        url,
        websocket_headers.to_vec(),
        cache_session_id,
        account_id,
        signal,
        options.websocket_connect_timeout_ms,
    )
    .await?;
    let request_body = if use_cached_context && let Some((session, account)) = &slot {
        let mut state = lock_state();
        find_entry_mut(&mut state, session, account).map_or_else(
            || full_body.clone(),
            |entry| build_cached_websocket_request_body(entry, full_body),
        )
    } else {
        full_body.clone()
    };

    // The transport bookkeeping of the attempt, upstream's stats block. The
    // reads precompute before the lock so the guard holds only the writes.
    if let Some(session_id) = cache_session_id {
        let store_true = request_body.get("store") == Some(&json!(true));
        let input_items = u64::try_from(
            request_body
                .get("input")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
        )
        .unwrap_or(u64::MAX);
        let previous_response_id = request_body
            .get("previous_response_id")
            .and_then(Value::as_str)
            .filter(|previous| !previous.is_empty());
        let mut state = lock_state();
        let session_stats = get_or_create_stats(&mut state, session_id);
        session_stats.requests += 1;
        if reused {
            session_stats.connections_reused += 1;
        } else {
            session_stats.connections_created += 1;
        }
        if use_cached_context {
            session_stats.cached_context_requests += 1;
        }
        if store_true {
            session_stats.store_true_requests += 1;
        }
        session_stats.last_input_items = input_items;
        if let Some(previous) = previous_response_id {
            session_stats.delta_requests += 1;
            session_stats.last_delta_input_items = Some(input_items);
            session_stats.last_previous_response_id = Some(previous.to_owned());
        } else {
            session_stats.full_context_requests += 1;
            session_stats.last_delta_input_items = None;
            session_stats.last_previous_response_id = None;
        }
        drop(state);
    }

    // The connection-scoped continuation keeps the socket usable across
    // turns where `store: true` is rejected, so the cached-context rewrite
    // is the only path that changes the request shape.
    let socket_slot: SocketSlot = Arc::new(Mutex::new(None));
    let outcome: Result<(), StreamFailure> = async {
        let Some(mut connection) = connection else {
            return Err(StreamFailure::plain(
                "WebSocket connection is not available",
            ));
        };
        let mut frame = Map::new();
        frame.insert("type".to_owned(), json!("response.create"));
        frame.extend(
            request_body
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
        if let Err(error) = connection
            .send(WebSocketMessage::Text(Value::Object(frame).to_string()))
            .await
        {
            *socket_slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(connection);
            return Err(StreamFailure::plain(error.to_string()));
        }
        let pump = MappedPump {
            source: PumpSource::WebSocket(WebSocketSide {
                slot: Arc::clone(&socket_slot),
                connection: Some(connection),
                idle_timeout_ms: idle_timeout_ms.filter(|timeout_ms| *timeout_ms > 0),
                saw_completion: false,
            }),
            shared: Arc::clone(shared),
            forward: events.clone(),
            start_partial: output.clone(),
            finished: false,
        };
        run_mapped_stream(pump, output, events, model, stream_options).await
    }
    .await;

    // Continuation bookkeeping before the release, upstream's try/catch body:
    // success stores the replay state, every failure clears it, and an abort
    // at the end of the stream also drops the connection.
    let mut keep_connection = true;
    let mut outcome = outcome;
    if matches!(&outcome, Ok(())) {
        if signal.is_cancelled() {
            keep_connection = false;
        } else if use_cached_context
            && let Some((session, account)) = &slot
            && let Some(response_id) = output
                .response_id
                .clone()
                .filter(|response_id| !response_id.is_empty())
        {
            match build_websocket_continuation_items(model, output, grammar_tool_input_properties) {
                Ok(items) => {
                    let mut state = lock_state();
                    if let Some(entry) = find_entry_mut(&mut state, session, account) {
                        entry.continuation = Some(CachedWebSocketContinuationState {
                            last_request_body: full_body.clone(),
                            last_response_id: response_id,
                            last_response_items: items,
                        });
                    }
                }
                Err(failure) => {
                    clear_cached_websocket_continuation(session, account);
                    keep_connection = false;
                    outcome = Err(failure);
                }
            }
        }
    } else {
        if let Some((session, account)) = &slot {
            clear_cached_websocket_continuation(session, account);
        }
        keep_connection = false;
    }
    let released = socket_slot
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    release_websocket(slot, released, keep_connection);
    outcome
}

/// Stream a simple request, upstream's `streamSimple`: the pi reasoning level
/// clamps to the model's supported levels and spends an effort, with the
/// clamped-off state spending none.
#[must_use]
pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    if options
        .and_then(|options| options.api_key.as_deref())
        .is_none_or(str::is_empty)
    {
        return setup_error_stream(
            model,
            &format!("No API key for provider: {}", model.provider.0),
        );
    }

    let mut base = OpenAiCodexResponsesOptions::from(build_base_options(
        model,
        context,
        options,
        options.and_then(|options| options.api_key.as_deref()),
    ));
    base.tool_choice = options
        .and_then(|options| options.tool_choice)
        .map(|choice| match choice {
            ToolChoice::Auto => CodexToolChoice::Auto,
            ToolChoice::None => CodexToolChoice::None,
        });
    base.reasoning_effort = options
        .and_then(|options| options.reasoning)
        .map(|reasoning| clamp_thinking_level(model, thinking_model_level(reasoning)))
        .filter(|level| *level != ModelThinkingLevel::Off);
    stream(model, context, Some(&base))
}
