//! The assistant stream runner, ported from upstream
//! `src/harness/execution/assistant.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! [`stream_harness_assistant`] runs one already-approved provider request
//! through its lifecycle — context transform, provider-message conversion,
//! the request, the observer, and the after-response hook — without
//! mutating the caller's message list. The request hook restates
//! upstream's `AssistantMessageEventStream | Promise<...>` union as a
//! boxed future; sync callers wrap their stream in a ready future.
//!
//! Porting restatements: the observer's optional async callbacks restate
//! as one trait whose methods return boxed futures; upstream's thrown
//! protocol violations restate as [`AssistantStreamError::Protocol`] with
//! the upstream messages verbatim; and `signal: context.abortSignal`
//! restates as a cancellation token linked to the context's chord signal —
//! the link task cancels the token when the signal aborts and exits when
//! either side settles, so it lives as long as the drive pass the context
//! belongs to.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::PoisonError;

use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, BoxedFuture, Context as AiContext, Message, Model,
    OnPayload, OnResponse, SimpleStreamOptions, Tool, TransportOptions,
};
use pi_ai::utils::assistant_message_frame::event_type_name;
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::Value as JsonValue;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{event_partial, stream_reasoning};
use crate::harness::context::{Context, get_telemetry_context};
use crate::harness::gate::AbortRequested;
use crate::harness::session::types::{SettledAssistantMessage, SettledStopReason};
use crate::harness::types::AgentHarnessStreamOptions;
use crate::types::{AgentMessage, ThinkingLevel};

/// HTTP response metadata captured before the provider response body is
/// consumed, upstream's `AssistantResponseMetadata`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssistantResponseMetadata {
    /// The HTTP status code.
    pub status: Option<u16>,
    /// The response headers.
    pub headers: Option<BTreeMap<String, String>>,
}

/// The captured-metadata cell the response hook writes and the after-
/// response callback reads at settlement time.
type MetadataCell = Arc<Mutex<Option<AssistantResponseMetadata>>>;

/// Process-local lifecycle observer for one assistant stream, upstream's
/// `AssistantStreamObserver`.
///
/// `start` and `update` receive event-time copies of the live accumulator,
/// upstream's `{ ...event.partial }` spread at the call sites.
pub trait AssistantStreamObserver: Send + Sync {
    /// The stream opened.
    fn start(
        &self,
        message: AssistantMessage,
        event: &AssistantMessageEvent,
        context: &Context,
    ) -> BoxedFuture<'_, ()>;

    /// A partial update arrived.
    fn update(
        &self,
        message: AssistantMessage,
        event: &AssistantMessageEvent,
        context: &Context,
    ) -> BoxedFuture<'_, ()>;

    /// The stream settled; `message` is the final (possibly replaced)
    /// message.
    fn end(&self, message: &SettledAssistantMessage, context: &Context) -> BoxedFuture<'_, ()>;
}

/// The transformable request context, upstream's inline
/// `{ messages, systemPrompt }` object the transform hook receives and
/// returns.
#[derive(Clone, Debug, PartialEq)]
pub struct AssistantRequestContext {
    /// The messages the request builds from.
    pub messages: Vec<AgentMessage>,
    /// The system prompt the request carries.
    pub system_prompt: String,
}

/// The context-transform hook, upstream's `transformContext`.
pub type TransformContextHook = Arc<
    dyn for<'a> Fn(
            AssistantRequestContext,
            &'a Context,
        ) -> BoxedFuture<
            'a,
            Result<AssistantRequestContext, Box<dyn std::error::Error + Send + Sync>>,
        > + Send
        + Sync,
>;

/// The provider-message conversion hook, upstream's `toProviderMessages`.
pub type ToProviderMessages = Arc<
    dyn for<'a> Fn(&'a [AgentMessage], &'a Context) -> BoxedFuture<'a, Vec<Message>> + Send + Sync,
>;

/// The request-payload hook, upstream's `beforePayload`.
///
/// The context rides by value — the pi-ai `OnPayload` seam the hook
/// installs into requires a `'static` future, and chord's context handle
/// is a cheap clone.
pub type BeforePayloadHook =
    Arc<dyn Fn(JsonValue, Model, Context) -> BoxedFuture<'static, Option<JsonValue>> + Send + Sync>;

/// The after-response hook, upstream's `afterResponse`.
///
/// An [`AbortRequested`] rejection keeps the raw settlement; any other
/// failure propagates. The context rides by value — the metadata-binding
/// wrapper below re-wraps the hook into a `'static` future, and chord's
/// context handle is a cheap clone, the [`BeforePayloadHook`] precedent.
pub type AfterResponseHook = Arc<
    dyn Fn(
            SettledAssistantMessage,
            AssistantResponseMetadata,
            Context,
        ) -> BoxedFuture<
            'static,
            Result<SettledAssistantMessage, Box<dyn std::error::Error + Send + Sync>>,
        > + Send
        + Sync,
>;

/// The after-response hook bound to its request's captured metadata,
/// upstream's `(message, afterContext) => afterResponse(message, metadata,
/// afterContext)` closure.
///
/// The metadata reads at settlement time, so a response hook firing
/// mid-stream wins.
pub type AfterResponseCallback = Arc<
    dyn Fn(
            SettledAssistantMessage,
            Context,
        ) -> BoxedFuture<
            'static,
            Result<SettledAssistantMessage, Box<dyn std::error::Error + Send + Sync>>,
        > + Send
        + Sync,
>;

/// The provider-request hook, upstream's `request`.
///
/// Upstream types the return as the
/// `AssistantMessageEventStream | Promise<...>` union; the boxed future is
/// that union, and sync callers wrap their stream in a ready future — the
/// payload and response hooks fire inside it, matching upstream's async
/// request bodies.
pub type AssistantRequestHook = Arc<
    dyn for<'a> Fn(
            &'a AiContext,
            SimpleStreamOptions,
            &'a Context,
        ) -> BoxedFuture<'a, AssistantMessageEventStream>
        + Send
        + Sync,
>;

/// Executable inputs for one already-approved assistant provider request,
/// upstream's `HarnessAssistantStreamConfig`.
pub struct HarnessAssistantStreamConfig {
    /// The model the request streams from.
    pub model: Model,
    /// The system prompt the request carries (before transformation).
    pub system_prompt: String,
    /// The provider tool surface the request advertises.
    pub tools: Option<Vec<Tool>>,
    /// The pi thinking level for the request; `off` sends no reasoning
    /// field.
    pub thinking_level: ThinkingLevel,
    /// The curated request options the harness snapshotted for the turn.
    pub stream_options: AgentHarnessStreamOptions,
    /// The context-transform hook, upstream's `transformContext`.
    pub transform_context: Option<TransformContextHook>,
    /// The provider-message conversion hook, upstream's `toProviderMessages`.
    pub to_provider_messages: ToProviderMessages,
    /// The request-payload hook, upstream's `beforePayload`.
    pub before_payload: Option<BeforePayloadHook>,
    /// The after-response hook, upstream's `afterResponse`.
    pub after_response: Option<AfterResponseHook>,
    /// The provider-request hook, upstream's `request`.
    pub request: AssistantRequestHook,
    /// The lifecycle observer, upstream's `observer`.
    pub observer: Arc<dyn AssistantStreamObserver>,
}

impl std::fmt::Debug for HarnessAssistantStreamConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The hooks and the observer do not debug; the declarative surface
        // does, and the non-exhaustive finish names their presence.
        f.debug_struct("HarnessAssistantStreamConfig")
            .field("model", &self.model)
            .field("system_prompt", &self.system_prompt)
            .field("tools", &self.tools)
            .field("thinking_level", &self.thinking_level)
            .field("stream_options", &self.stream_options)
            .field("transform_context", &self.transform_context.is_some())
            .field("to_provider_messages", &())
            .field("before_payload", &self.before_payload.is_some())
            .field("after_response", &self.after_response.is_some())
            .field("request", &())
            .field("observer", &())
            .finish_non_exhaustive()
    }
}

/// What streaming one assistant response surfaces, the restatement of
/// upstream's thrown errors in `assistant.ts`.
#[derive(Debug)]
pub enum AssistantStreamError {
    /// The stream broke its protocol: a second `start`, or an update or
    /// `done` before `start`. The message is upstream's verbatim.
    Protocol(String),
    /// A caller hook failed: `transform_context`, or an `after_response`
    /// failure that is not an [`AbortRequested`].
    Hook(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for AssistantStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(message) => f.write_str(message),
            Self::Hook(error) => std::fmt::Display::fmt(&error, f),
        }
    }
}

impl std::error::Error for AssistantStreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(_) => None,
            Self::Hook(error) => Some(error.as_ref()),
        }
    }
}

/// The cancellation token the request options carry for the context's
/// abort signal, upstream's `signal: context.abortSignal`.
///
/// pi-ai's request cancellation is the workspace `CancellationToken` while
/// the harness context carries chord's `AbortSignal`; the link task
/// cancels the token when the signal aborts and exits when the token is
/// cancelled first, so a request that finishes before the signal aborts
/// leaves the parked task to the drive pass's end.
fn request_signal(context: &Context) -> Option<CancellationToken> {
    let signal = context.abort_signal()?;
    let token = CancellationToken::new();
    let linked = token.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = signal.wait() => linked.cancel(),
            () = linked.cancelled() => {},
        }
    });
    Some(token)
}

/// Build the provider request options from the curated snapshot,
/// upstream's `createRequestOptions`.
fn create_request_options(
    config: &HarnessAssistantStreamConfig,
    capture_metadata: impl Fn(AssistantResponseMetadata) + Send + Sync + 'static,
    context: &Context,
) -> SimpleStreamOptions {
    let options = &config.stream_options;
    SimpleStreamOptions {
        transport_options: TransportOptions {
            http_client: None,
            signal: request_signal(context),
            on_payload: config.before_payload.clone().map(|hook| {
                let context = context.clone();
                OnPayload::new(move |payload, model| {
                    let context = context.clone();
                    hook(payload, model, context)
                })
            }),
            on_response: Some(OnResponse::new(move |response, _model| {
                capture_metadata(AssistantResponseMetadata {
                    status: Some(response.status),
                    headers: Some(response.headers),
                });
                Box::pin(async {})
            })),
        },
        headers: options.headers.clone().map(|headers| {
            headers
                .into_iter()
                .map(|(key, value)| (key, Some(value)))
                .collect()
        }),
        timeout_ms: options.timeout_ms,
        max_retries: options.max_retries,
        max_retry_delay_ms: options.max_retry_delay_ms,
        metadata: options.metadata.clone(),
        transport: options.transport,
        cache_retention: options.cache_retention,
        deferred: options.deferred.clone(),
        reasoning: stream_reasoning(config.thinking_level),
        telemetry_context: Some(get_telemetry_context(context)),
        ..SimpleStreamOptions::default()
    }
}

/// Consume one assistant stream through the observer and the after-response
/// hook, upstream's `consumeAssistantStream`.
///
/// The stream protocol binds the lifecycle: `start` exactly once, updates
/// only after it, `done` only after it. An `error` event never throws —
/// its settlement reaches the caller through the stream result.
///
/// # Errors
/// [`AssistantStreamError::Protocol`] on the violations above;
/// [`AssistantStreamError::Hook`] when the after-response hook fails with
/// anything but an [`AbortRequested`], whose cancellation the runner awaits
/// and whose raw settlement stands.
///
/// # Panics
/// When the settled message's stop reason is `pending` — the stream
/// resolves only from `done` and `error` terminations, so a pending
/// settlement is a stream-protocol violation the runner cannot encode.
pub async fn consume_assistant_stream(
    stream: &AssistantMessageEventStream,
    observer: &dyn AssistantStreamObserver,
    after_response: Option<AfterResponseCallback>,
    context: &Context,
) -> Result<SettledAssistantMessage, AssistantStreamError> {
    let mut started = false;
    while let Some(event) = stream.next().await {
        match &event {
            AssistantMessageEvent::Start { partial } => {
                if started {
                    return Err(AssistantStreamError::Protocol(
                        "Assistant message stream emitted more than one start event".to_owned(),
                    ));
                }
                started = true;
                observer.start(partial.clone(), &event, context).await;
            }
            AssistantMessageEvent::Done { .. } if !started => {
                return Err(AssistantStreamError::Protocol(
                    "Assistant message stream emitted done before start".to_owned(),
                ));
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {}
            event => {
                if !started {
                    return Err(AssistantStreamError::Protocol(format!(
                        "Assistant message stream emitted {} before start",
                        event_type_name(event)
                    )));
                }
                if let Some(partial) = event_partial(event) {
                    observer.update(partial.clone(), event, context).await;
                }
            }
        }
    }

    let settled_message = stream.result().await;
    #[expect(
        clippy::expect_used,
        reason = "the stream resolves only from done and error terminations, whose reasons are never pending; a pending settlement is the stream protocol's violation"
    )]
    let stop_reason = SettledStopReason::from_stop_reason(settled_message.stop_reason)
        .expect("the assistant stream settles only from done or error terminations");
    let settled = SettledAssistantMessage {
        message: settled_message,
        stop_reason,
    };
    let mut final_message = settled;
    if let Some(after_response) = after_response {
        match after_response(final_message.clone(), context.clone()).await {
            Ok(replaced) => final_message = replaced,
            Err(error) => {
                if let Some(abort) = error.downcast_ref::<AbortRequested>() {
                    let mut cancellation = abort.cancellation.clone();
                    drop(error);
                    // The cancellation settles when the abort work sends or
                    // the sender drops; either way the wait is done.
                    let _ = cancellation.changed().await;
                } else {
                    return Err(AssistantStreamError::Hook(error));
                }
            }
        }
    }
    observer.end(&final_message, context).await;
    Ok(final_message)
}

/// Stream one assistant response without mutating the caller's message
/// list, upstream's `streamHarnessAssistant`.
///
/// # Errors
/// [`AssistantStreamError::Hook`] when the context-transform hook fails,
/// and whatever [`consume_assistant_stream`] surfaces.
pub async fn stream_harness_assistant(
    messages: &[AgentMessage],
    config: &HarnessAssistantStreamConfig,
    context: &Context,
) -> Result<SettledAssistantMessage, AssistantStreamError> {
    let mut request_context = AssistantRequestContext {
        messages: messages.to_vec(),
        system_prompt: config.system_prompt.clone(),
    };
    if let Some(transform_context) = &config.transform_context {
        request_context = transform_context(request_context, context)
            .await
            .map_err(AssistantStreamError::Hook)?;
    }

    let provider_messages = (config.to_provider_messages)(&request_context.messages, context).await;
    let ai_context = AiContext {
        system_prompt: Some(request_context.system_prompt.clone()),
        messages: provider_messages,
        tools: config.tools.clone(),
    };

    let metadata_cell: MetadataCell = Arc::new(Mutex::new(None));
    let stream = (config.request)(
        &ai_context,
        create_request_options(
            config,
            {
                let cell = Arc::clone(&metadata_cell);
                move |metadata| {
                    *cell.lock().unwrap_or_else(PoisonError::into_inner) = Some(metadata);
                }
            },
            context,
        ),
        context,
    )
    .await;

    // The after-response callback reads the captured metadata at settlement
    // time, upstream's late read of the `metadata` variable.
    let after_response: Option<AfterResponseCallback> = config.after_response.clone().map_or_else(
        || None,
        |hook| {
            let cell = Arc::clone(&metadata_cell);
            let callback: AfterResponseCallback =
                Arc::new(move |message: SettledAssistantMessage, context: Context| {
                    let metadata = cell
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .clone()
                        .unwrap_or_default();
                    hook(message, metadata, context)
                });
            Some(callback)
        },
    );
    consume_assistant_stream(&stream, config.observer.as_ref(), after_response, context).await
}

#[cfg(test)]
mod tests;
