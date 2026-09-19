//! Shared normalization for provider HTTP error objects, ported from
//! `packages/ai/src/utils/error-body.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Endpoints behind a proxy or gateway may return a non-2xx response whose
//! body the provider SDK cannot fold into `error.message`. The SDK error
//! object still carries the HTTP status and the raw or parsed body, but under
//! SDK-specific field names, so a catch block reading only `error.message`
//! drops the body and surfaces opaque messages like
//! `"403 status code (no body)"`.
//!
//! [`normalize_provider_error`] probes the known SDK field shapes — Mistral
//! (`statusCode` / `body`), `openai` (`status` / `error`),
//! `@google/genai` (`status`), and AWS Bedrock (`$metadata.httpStatusCode` /
//! `$response`) — and returns a struct each provider composes into its
//! display string. The `messageCarriesBody` flag captures the Anthropic /
//! `@google/genai` happy path where the SDK already folded the body into the
//! message, so providers preserve it without double-printing.
//!
//! Porting restatement: TypeScript probes duck-typed fields on an `Error`
//! instance. The Rust port takes the probed fields as typed input
//! ([`SdkError`]) — the client seam maps its concrete errors onto it, and the
//! JS-runtime sniffing (`pipe` method, class-instance prototype checks) is
//! carried by the [`ErrorBody`] variants: a readable stream or a wrapper
//! class instance arrives as [`ErrorBody::Unreadable`], never as
//! serializable data.

use serde_json::Value;

/// The body-size cap upstream applies to extracted bodies.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// A normalized provider HTTP error: the extracted status and body, the
/// message, and whether the message already carries the body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizedProviderError {
    /// HTTP status code, when one could be extracted from the SDK error
    /// object.
    pub status: Option<u16>,
    /// Raw HTTP body reason, already trimmed and truncated to the cap.
    pub body: Option<String>,
    /// `error.message`, or `safeJsonStringify(error)` for a non-`Error`
    /// throw.
    pub message: String,
    /// True when `message` already contains the body (no separate body to
    /// add).
    pub message_carries_body: bool,
}

/// The body a SDK error field can carry: a raw string, a parsed JSON body,
/// or an unreadable stream or wrapper instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorBody {
    /// A raw body string, upstream's string-typed `body` / `$response.body`.
    Text(String),
    /// A parsed body value, upstream's object-typed `error` / `$response.body`.
    Parsed(Value),
    /// A response stream or SDK wrapper class instance: present, but never
    /// serialized as a body.
    Unreadable,
}

/// The `$response` field of a Bedrock-shaped error, upstream's
/// `{ statusCode, body }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SdkResponse {
    /// The HTTP status code.
    pub status_code: Option<u16>,
    /// The response body.
    pub body: Option<ErrorBody>,
}

/// The SDK error shapes the normalizer probes, upstream's `SdkErrorShape`.
///
/// Field names mirror the SDK fields they stand in for.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SdkError {
    /// `error.message`.
    pub message: String,
    /// Mistral's `statusCode`.
    pub status_code: Option<u16>,
    /// The `openai` / `@google/genai` `status`.
    pub status: Option<u16>,
    /// Mistral's raw `body` string.
    pub body: Option<String>,
    /// The `openai` SDK's parsed-body `error` field.
    pub error: Option<ErrorBody>,
    /// Bedrock's `$metadata.httpStatusCode`.
    pub metadata_http_status_code: Option<u16>,
    /// Bedrock's `$response`.
    pub response: Option<SdkResponse>,
}

impl SdkError {
    /// A non-`Error` thrown value: its message is the JSON stringification,
    /// upstream's `safeJsonStringify(error)` fallback.
    #[must_use]
    pub fn from_thrown(value: &Value) -> Self {
        Self {
            message: safe_json_stringify(value),
            ..Self::default()
        }
    }
}

/// Probe the HTTP status, first numeric hit wins, in SDK-field order:
/// `statusCode` (Mistral) → `status` (`openai`, `@google/genai`) →
/// `$metadata.httpStatusCode` (Bedrock) → `$response.statusCode` (Bedrock).
fn extract_status(error: &SdkError) -> Option<u16> {
    error
        .status_code
        .or(error.status)
        .or(error.metadata_http_status_code)
        .or_else(|| {
            error
                .response
                .as_ref()
                .and_then(|response| response.status_code)
        })
}

/// Probe the raw body reason, first usable hit wins, in SDK-field order:
/// `body` string (Mistral) → `error` parsed JSON body object (`openai` SDK's
/// `this.error`) → `$response.body` (Bedrock). Empty objects and unread
/// response streams are treated as no body so they do not surface as `"{}"`
/// or serialized stream internals. The chosen body is truncated to the cap.
fn extract_body(error: &SdkError) -> Option<String> {
    let body_text = pick_body_text(error)?;
    let trimmed = body_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS))
}

fn pick_body_text(error: &SdkError) -> Option<String> {
    if let Some(body) = &error.body {
        return Some(body.clone());
    }
    if let Some(ErrorBody::Parsed(error)) = &error.error {
        if is_plain_non_empty_object(error) {
            return Some(safe_json_stringify(error));
        }
        return None;
    }
    if let Some(response) = &error.response {
        match &response.body {
            Some(ErrorBody::Text(body)) => return Some(body.clone()),
            Some(ErrorBody::Parsed(parsed)) => {
                if is_plain_non_empty_object(parsed) {
                    return Some(safe_json_stringify(parsed));
                }
            }
            // An unread stream carries no body text; serializing its
            // internals would replace the real message with noise.
            Some(ErrorBody::Unreadable) => return None,
            None => {}
        }
    }
    None
}

/// Only a plain, non-empty object counts as an HTTP body. SDK error fields
/// can hold class instances instead of parsed bodies — stringifying one
/// produced garbage like `{"_events":...}` as the "body", which then replaced
/// `error.message`, where the SDK puts the real deserialized exception text.
/// A class instance yields no body, `messageCarriesBody` stays true, and the
/// real message survives. The parsed-JSON bodies this function accepts are
/// plain objects by construction.
fn is_plain_non_empty_object(value: &Value) -> bool {
    value.as_object().is_some_and(|object| !object.is_empty())
}

/// Normalize a non-`Error` thrown value: its message is the JSON
/// stringification and it carries no status or body, upstream's early
/// return for values that are not `Error` instances.
#[must_use]
pub fn normalize_thrown_value(value: &Value) -> NormalizedProviderError {
    NormalizedProviderError {
        status: None,
        body: None,
        message: safe_json_stringify(value),
        message_carries_body: false,
    }
}

/// Normalize a provider SDK error into the struct providers compose their
/// display string from.
#[must_use]
pub fn normalize_provider_error(error: SdkError) -> NormalizedProviderError {
    let status = extract_status(&error);
    let body = extract_body(&error);
    let message_carries_body = match &body {
        Some(body) => error.message.contains(body.as_str()),
        None => true,
    };
    NormalizedProviderError {
        status,
        body,
        message: error.message,
        message_carries_body,
    }
}

/// Compose a display string from a normalized error.
///
/// When the message already carries the body (Anthropic / `@google/genai`
/// happy path) or no body/status was extracted, the message is returned
/// unchanged. Otherwise the status and body are surfaced, with an optional
/// provider prefix.
///
/// - no prefix: `"<status>: <body>"`
/// - prefix:    `"<prefix> (<status>): <body>"`
#[must_use]
pub fn format_provider_error(norm: &NormalizedProviderError, prefix: Option<&str>) -> String {
    if norm.message_carries_body || norm.status.is_none() || norm.body.is_none() {
        return match (prefix, norm.status) {
            (Some(prefix), Some(status)) => format!("{prefix} ({status}): {}", norm.message),
            _ => norm.message.clone(),
        };
    }
    let body = norm.body.as_deref().unwrap_or_default();
    let status = norm.status.unwrap_or_default();
    prefix.map_or_else(
        || format!("{status}: {body}"),
        |prefix| format!("{prefix} ({status}): {body}"),
    )
}

/// Truncate long error text at `max_chars`, noting the dropped length.
#[must_use]
pub fn truncate_error_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    let dropped = text.chars().count() - max_chars;
    format!("{kept}... [truncated {dropped} chars]")
}

/// Stringify a JSON value, falling back to its display form when serialization fails.
///
/// Upstream's `safeJsonStringify` also maps `undefined` to its argument's
/// string form; serde_json values have no undefined, so the fallthrough only
/// guards serialization.
#[must_use]
pub fn safe_json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
}
