//! The helpers the wire-API adapters share: the fresh accumulator, the
//! settled setup-failure stream, the seam-error mapping, and the header
//! upsert.
//!
//! The stream-shell side of the sharing — the spawned `stream` wrapper, the
//! retried dispatch, the response-hook dispatch, the SSE transport-error
//! fold, and the streamed block-close bookkeeping — lives here too.
//!
//! Each upstream adapter file carried its own copy because the SDK owned the
//! plumbing; the port keeps one substrate per helper so the duplication gate
//! pins a single implementation.

use std::sync::Arc;

use crate::http::client::{HttpClient, HttpError, HttpRequest, HttpResponse, read_body_text};
use crate::types::{
    AssistantBlock, AssistantMessage, AssistantMessageEvent, BoxedFuture, Context, Model,
    ProviderId, ProviderResponse, StopReason, TextContent, ThinkingContent, TransportOptions,
};
use crate::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use crate::utils::headers::headers_to_record;
use crate::utils::provider_retry::{
    ProviderRequestError, ProviderRetryOptions, retry_provider_request,
};

/// The options shapes that carry a [`TransportOptions`] bundle, the seam the
/// shared stream shell re-checks the cancellation signal through.
///
/// The option structs spell the bundle out field by field; this trait is the
/// one shared accessor.
pub trait TransportCarrier {
    /// The transport seam of these options.
    fn transport_options(&self) -> &TransportOptions;
}

/// The fresh accumulator a stream starts from, upstream's `output` in every
/// adapter: zeroed usage and `stopReason: "pending"`.
#[must_use]
pub fn initial_output(model: &Model) -> AssistantMessage {
    initial_output_with_thinking_level(model, None)
}

/// The fresh accumulator a stream starts from, with an optional
/// provider-native thinking level pre-seeded (Anthropic's mid-conversation
/// effort rides there).
#[must_use]
pub fn initial_output_with_thinking_level(
    model: &Model,
    provider_thinking_level: Option<String>,
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
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

/// The stream-setup failure as a settled error stream, upstream's
/// synchronous `streamSimple` throw encoded per the stream contract.
#[must_use]
pub fn setup_error_stream(model: &Model, message: &str) -> AssistantMessageEventStream {
    let events = assistant_message_event_stream();
    let failure = std::io::Error::other(message.to_owned());
    let failing = crate::api::lazy::setup_error_message(model, &failure);
    events.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: failing.clone(),
    });
    events.end(Some(&failing));
    events
}

/// Map a transport failure to the provider-request error the retry seam
/// classifies, the shared shape of every adapter's `providerErrorFromHttp`.
///
/// Aborts keep their identity, timeouts and transport failures carry the
/// message, invalid URLs are named.
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

/// The retry options an adapter hands [`retry_provider_request`], upstream's
/// `ProviderRetryOptions` construction from the shared option fields.
#[must_use]
pub fn retry_options(
    max_retries: Option<u32>,
    max_retry_delay_ms: Option<u64>,
    signal: tokio_util::sync::CancellationToken,
) -> ProviderRetryOptions {
    ProviderRetryOptions {
        max_retries: max_retries.unwrap_or(0),
        max_retry_delay_ms,
        signal: Some(signal),
        random: None,
    }
}

/// Insert or replace a header by case-insensitive name, the shared shape of
/// every adapter's header merge.
pub fn upsert_header(headers: &mut Vec<(String, String)>, name: &str, value: &str) {
    if let Some(entry) = headers
        .iter_mut()
        .find(|(existing, _)| existing.eq_ignore_ascii_case(name))
    {
        value.clone_into(&mut entry.1);
    } else {
        headers.push((name.to_owned(), value.to_owned()));
    }
}

/// The `No API key for provider: ...` wording the adapters share.
#[must_use]
pub fn missing_api_key_message(provider: &ProviderId) -> String {
    format!("No API key for provider: {}", provider.0)
}

/// Append a header only when no same-named (case-insensitive) header exists,
/// the append-only shape of the adapters' credential headers.
pub fn push_header_if_absent(headers: &mut Vec<(String, String)>, name: &str, value: String) {
    if !headers
        .iter()
        .any(|(existing, _)| existing.eq_ignore_ascii_case(name))
    {
        headers.push((name.to_owned(), value));
    }
}

/// Spawn a stream's run task, the shared shape of every adapter's `stream`
/// export.
///
/// The event stream is created up front, the task clones the model and
/// context, seeds the accumulator through `initial`, and on failure settles
/// the accumulator as the stream's terminal error event (`Aborted` when the
/// signal cancelled, `Error` otherwise).
pub fn spawn_adapter_stream<O, I, F>(
    model: &Model,
    context: &Context,
    options: Option<O>,
    initial: I,
    run: F,
) -> AssistantMessageEventStream
where
    O: Default + Send + TransportCarrier + 'static,
    I: FnOnce(&Model, &O) -> AssistantMessage + Send + 'static,
    F: for<'a> FnOnce(
            &'a Model,
            &'a Context,
            &'a O,
            &'a mut AssistantMessage,
            &'a AssistantMessageEventStream,
        ) -> BoxedFuture<'a, Result<(), String>>
        + Send
        + 'static,
{
    let events = assistant_message_event_stream();
    let forward = events.clone();
    let model = model.clone();
    let context = context.clone();
    tokio::spawn(async move {
        let options = options.unwrap_or_default();
        let mut output = initial(&model, &options);
        match run(&model, &context, &options, &mut output, &forward).await {
            Ok(()) => {
                // The done event already settled the final result.
                forward.end(None);
            }
            Err(message) => {
                let signal = options.transport_options().signal();
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

/// The response hook dispatch, upstream's `onResponse`: the settled response
/// rides as `{ status, headers }` with the header list folded to a record,
/// before the body is consumed.
///
/// The response is passed by fields, not by reference: the body stream is
/// not `Sync`, so a `&HttpResponse` cannot cross the hook's await.
pub async fn call_response_hook(
    transport_options: &TransportOptions,
    model: &Model,
    status: u16,
    headers: &[(String, String)],
) {
    if let Some(hook) = &transport_options.on_response {
        hook.call(
            ProviderResponse {
                status,
                headers: headers_to_record(
                    headers
                        .iter()
                        .map(|(name, value)| (name.as_str(), value.as_str())),
                ),
            },
            model.clone(),
        )
        .await;
    }
}

/// Dispatch a request through the shared provider retry policy, the seam-side
/// port of upstream's per-API `retryXRequest(() => fetch(...))`.
///
/// One retried execution, a non-2xx response folded into the provider-request
/// error the retry seam classifies, then the response hook on the settled
/// response.
///
/// The pinned SDK folds the response body into its error message and throws
/// after exhausting retries; `error_message` owns that status/body fold per
/// backend.
///
/// # Errors
/// When retries are exhausted or the request is not a 2xx: the folded
/// error's message.
pub async fn dispatch_with_retry(
    http_client: Arc<dyn HttpClient>,
    request: HttpRequest,
    retry: &ProviderRetryOptions,
    error_message: fn(u16, &str) -> String,
    model: &Model,
    transport_options: &TransportOptions,
) -> Result<HttpResponse, String> {
    let response = retry_provider_request(
        || {
            let client = Arc::clone(&http_client);
            let request = request.clone();
            async move {
                let response = client
                    .execute(request)
                    .await
                    .map_err(provider_error_from_http)?;
                if !(200..300).contains(&response.status) {
                    let body_text = read_body_text(response.body).await.unwrap_or_default();
                    return Err(ProviderRequestError::new(
                        Some(response.status),
                        Some(headers_to_record(
                            response
                                .headers
                                .iter()
                                .map(|(name, value)| (name.as_str(), value.as_str())),
                        )),
                        error_message(response.status, &body_text),
                    ));
                }
                Ok(response)
            }
        },
        retry,
    )
    .await
    .map_err(|error| error.message)?;

    call_response_hook(transport_options, model, response.status, &response.headers).await;

    Ok(response)
}

/// The SSE consumption failure message, the transport-error fold the stream
/// loops map `HttpError` through.
///
/// An abort keeps the wording every failure path matches on, other failures
/// carry the error's display form.
#[must_use]
pub fn sse_error_message(error: HttpError) -> String {
    match error {
        HttpError::Aborted => "Request was aborted".to_owned(),
        other => other.to_string(),
    }
}

/// Close the open text/thinking block, emitting its `*_end` event.
pub fn close_current_block(
    current_block: &mut Option<bool>,
    output: &AssistantMessage,
    events: &AssistantMessageEventStream,
) {
    if let Some(open_thinking) = *current_block {
        let content_index = output.content.len() as u64 - 1;
        if open_thinking {
            let Some(AssistantBlock::Thinking(block)) = output.content.last() else {
                return;
            };
            events.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: block.thinking.clone(),
                partial: output.clone(),
            });
        } else {
            let Some(AssistantBlock::Text(block)) = output.content.last() else {
                return;
            };
            events.push(AssistantMessageEvent::TextEnd {
                content_index,
                content: block.text.clone(),
                partial: output.clone(),
            });
        }
        *current_block = None;
    }
}

/// Open a text/thinking block when the open block is not of that kind:
/// close whatever is open, push the fresh block, emit its `*_start` event,
/// and mark it open, upstream's block-switch handling.
pub fn open_stream_block(
    current_block: &mut Option<bool>,
    output: &mut AssistantMessage,
    events: &AssistantMessageEventStream,
    is_thinking: bool,
) {
    if *current_block == Some(is_thinking) {
        return;
    }
    close_current_block(current_block, output, events);
    if is_thinking {
        output
            .content
            .push(AssistantBlock::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            }));
        *current_block = Some(true);
        events.push(AssistantMessageEvent::ThinkingStart {
            content_index: output.content.len() as u64 - 1,
            partial: output.clone(),
        });
    } else {
        output.content.push(AssistantBlock::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
        *current_block = Some(false);
        events.push(AssistantMessageEvent::TextStart {
            content_index: output.content.len() as u64 - 1,
            partial: output.clone(),
        });
    }
}

/// The `ProviderStreams` forwarding every wire adapter repeats: `stream`
/// rebuilds the adapter options from the wire options and hands them to the
/// module's `stream`, `streamSimple` forwards the shared options verbatim.
///
/// `stream` and `stream_simple` are the module-local exports the macro is
/// used beside.
macro_rules! forward_provider_streams {
    ($streams:ident, $options:path) => {
        impl crate::types::ProviderStreams for $streams {
            fn stream(
                &self,
                model: &crate::types::Model,
                context: &crate::types::Context,
                options: Option<&crate::types::StreamOptions>,
            ) -> crate::utils::event_stream::AssistantMessageEventStream {
                let options = options.cloned().map(<$options>::from);
                stream(model, context, options.as_ref())
            }

            fn stream_simple(
                &self,
                model: &crate::types::Model,
                context: &crate::types::Context,
                options: Option<&crate::types::SimpleStreamOptions>,
            ) -> crate::utils::event_stream::AssistantMessageEventStream {
                stream_simple(model, context, options)
            }
        }
    };
}
pub(crate) use forward_provider_streams;
