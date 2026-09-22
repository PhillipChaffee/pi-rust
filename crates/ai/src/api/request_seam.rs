//! The wire-API request seam's shared mechanics, ported from the per-API
//! copies of upstream's transport helpers at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream's `packages/ai/src/api/*.ts` modules each imported the same
//! request-pipeline pieces — the `APIError.makeMessage` error text, the
//! `response.text()` body drain, the `HttpError` mapping into the retry
//! error, and the throwing-`streamSimple` setup failure. The port keeps them
//! in one `pub(crate)` module so every wire API rides the identical seam
//! without copy-paste.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::http::client::{HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse};
use crate::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, ProviderHeaders, StopReason,
    TransportOptions,
};
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::ProviderRequestError;

/// Whether a header map carries a non-empty value for `name`
/// (case-insensitive), upstream's `hasHeader`.
#[must_use]
pub fn has_header(headers: Option<&ProviderHeaders>, name: &str) -> bool {
    headers.is_some_and(|headers| {
        headers.iter().any(|(key, value)| {
            key.eq_ignore_ascii_case(name)
                && value
                    .as_deref()
                    .is_some_and(|value| !value.trim().is_empty())
        })
    })
}

/// The credential a request sends, upstream's `getClientApiKey`: the caller's
/// key, else `"unused"` when the headers carry their own authorization.
///
/// # Errors
/// A provider without any credential fails with the missing-key message.
pub fn get_client_api_key(
    provider: &str,
    api_key: Option<&str>,
    headers: Option<&ProviderHeaders>,
) -> Result<String, String> {
    if let Some(api_key) = api_key {
        return Ok(api_key.to_owned());
    }
    if has_header(headers, "authorization") || has_header(headers, "cf-aig-authorization") {
        return Ok("unused".to_owned());
    }
    Err(format!("No API key for provider: {provider}"))
}

/// Read a response body to text, the seam-side port of `response.text()`.
///
/// # Errors
/// The body stream's error; see [`HttpByteStream::next_chunk`].
pub async fn read_body_text(mut body: HttpByteStream) -> Result<String, HttpError> {
    let mut text = String::new();
    while let Some(chunk) = body.next_chunk().await? {
        text.push_str(&String::from_utf8_lossy(&chunk));
    }
    Ok(text)
}

/// The transport error a request failure maps to, upstream's
/// `APIConnectionError`/timeout/abort shape.
#[must_use]
pub fn provider_error_from_http(error: HttpError) -> ProviderRequestError {
    match error {
        HttpError::Aborted => ProviderRequestError::aborted(),
        HttpError::Timeout => ProviderRequestError::new(None, None, "request timed out"),
        HttpError::Transport(message) => ProviderRequestError::new(None, None, message),
        HttpError::InvalidUrl(url) => {
            ProviderRequestError::new(None, None, format!("invalid URL: {url}"))
        }
    }
}

/// The SDK-shaped error message of a non-2xx response, upstream's
/// `APIError.makeMessage`.
///
/// The message is `{status} {parsed body}` when the body parses,
/// `{status} {raw body}` when it does not, `{status} status code (no body)`
/// otherwise.
#[must_use]
pub fn sdk_error_message(status: u16, body: &str) -> String {
    match serde_json::from_str::<Value>(body) {
        Ok(parsed) => format!("{status} {parsed}"),
        Err(_) if !body.is_empty() => format!("{status} {body}"),
        Err(_) => format!("{status} status code (no body)"),
    }
}

/// Execute one request over the seam client and fold a non-2xx response into
/// the SDK-shaped provider error, upstream's `create(...).asResponse()` unit
/// inside `retryProviderRequest`.
///
/// The response body is parsed into `parsed_error_body` before it is dropped
/// so the caller's failure path can compose the message the pinned SDK
/// produced and, where the provider carries one, the raw-body metadata.
///
/// # Errors
/// The transport failure of the request, or the non-2xx response folded into
/// the SDK-shaped provider error.
pub async fn execute_checked_response(
    client: Arc<dyn HttpClient>,
    request: HttpRequest,
    parsed_error_body: Option<&std::sync::Mutex<Option<Value>>>,
) -> Result<HttpResponse, ProviderRequestError> {
    let response = client
        .execute(request)
        .await
        .map_err(provider_error_from_http)?;
    if !(200..300).contains(&response.status) {
        // The pinned SDK folds the body into its error message and throws
        // after exhausting retries.
        let body_text = read_body_text(response.body).await.unwrap_or_default();
        if let Some(slot) = parsed_error_body
            && let Ok(mut slot) = slot.lock()
        {
            *slot = serde_json::from_str(&body_text).ok();
        }
        return Err(ProviderRequestError::new(
            Some(response.status),
            Some(headers_to_record(
                response
                    .headers
                    .iter()
                    .map(|(name, value)| (name.as_str(), value.as_str())),
            )),
            sdk_error_message(response.status, &body_text),
        ));
    }
    Ok(response)
}

/// Fire the transport's response hook with the SDK-shaped response record,
/// upstream's `onResponse` await in the request pipeline.
///
/// The response rides in as its copied fields: the seam's [`HttpResponse`]
/// is not `Sync`, and the stream it carries has no reason to cross this
/// await.
pub async fn fire_response_hook(
    transport_options: &TransportOptions,
    status: u16,
    headers: &[(String, String)],
    model: Model,
) {
    let Some(hook) = &transport_options.on_response else {
        return;
    };
    let record = crate::types::ProviderResponse {
        status,
        headers: headers_to_record(
            headers
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
        ),
    };
    hook.call(record, model).await;
}

/// The failure stream a setup-time rejection returns: the error lands as the
/// stream's `error` event and the stream settles, the Rust shape of
/// upstream's throwing `streamSimple` prelude.
#[must_use]
pub fn setup_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let failure: std::io::Error = std::io::Error::other(message.to_owned());
    let failing = crate::api::lazy::setup_error_message(model, &failure);
    events.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failing.clone(),
    });
    events.end(Some(&failing));
    events
}

/// The fresh accumulator a stream starts from, upstream's `output`: zeroed
/// usage, `stopReason: "pending"`, the model's identity on the message, and
/// the wire-API id the stream reports.
#[must_use]
pub fn initial_output(
    model: &Model,
    api: crate::types::Api,
    provider_thinking_level: Option<String>,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api,
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level,
        diagnostics: None,
        usage: crate::types::Usage::default(),
        stop_reason: StopReason::Pending,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: crate::auth::resolve::now_ms(),
    }
}

/// Spawn the runtime task a wire-API `stream` returns live from, upstream's
/// per-API `stream` wrapper.
///
/// The caller already holds the live stream while the request runs on a
/// spawned task; the adapter's `run` closure receives the accumulator and the
/// stream handle, the `done` event settles the final result, and a failure
/// lands as the `error` event carrying the output's abort- or error-shaped
/// stop reason.
pub fn spawned_stream<S>(
    model: &Model,
    context: &Context,
    signal: CancellationToken,
    initial_output: AssistantMessage,
    run: S,
) -> AssistantMessageEventStream
where
    S: FnOnce(Model, Context, &mut AssistantMessage, AssistantMessageEventStream) -> PinnedRun
        + Send
        + 'static,
{
    let events = assistant_message_event_stream();
    let forward = events.clone();
    let model = model.clone();
    let context = context.clone();
    tokio::spawn(async move {
        let mut output = initial_output;
        match run(model, context, &mut output, forward.clone()).await {
            Ok(()) => {
                // The done event already settled the final result.
                forward.end(None);
            }
            Err(message) => {
                output.stop_reason = if signal.is_cancelled() {
                    StopReason::Aborted
                } else {
                    StopReason::Error
                };
                output.error_message = Some(message);
                forward.push(AssistantMessageEvent::Error {
                    reason: output.stop_reason,
                    error: output.clone(),
                });
                forward.end(None);
            }
        }
    });
    events
}

/// The boxed run future a wire-API adapter hands [`spawned_stream`]: the
/// accumulator borrow rides inside the boxed future's lifetime.
pub type PinnedRun<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;
