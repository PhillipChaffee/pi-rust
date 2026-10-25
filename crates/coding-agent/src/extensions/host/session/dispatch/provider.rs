//! Provider streams over the wire: a registered provider's `stream_simple`
//! closure opens one `call_provider` scope and relays the child's
//! `stream_event` frames into the host's event stream ([ADR 0010]).
//!
//! The request is the child's whole input: the model, the conversation
//! context, and the serializable simple-request options. Transport plumbing
//! and closures stay host-side; the child re-creates its own.
//!
//! [ADR 0010]: ../../../../../../docs/adr/0010-extension-host-wire-protocol.md

use std::sync::Arc;

use pi_ai::types::{
    AssistantMessage, AssistantMessageEvent, Context, Model, ProviderId, SimpleStreamOptions,
    StopReason, Usage,
};
use pi_ai::utils::event_stream::{
    AssistantMessageEventStream, create_assistant_message_event_stream,
};
use serde_json::{Value, json};

use super::super::super::Frame;
use super::super::super::context::HostCallError;
use super::super::call::{CallOutcome, CallSink, SignalScope};
use super::super::connection::SessionCore;

impl<T> SessionCore<T>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    /// Open one provider stream, driving the child from a background task.
    pub(super) fn stream_provider(
        self: &Arc<Self>,
        provider: &str,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let stream = create_assistant_message_event_stream();
        let request = match provider_request(model, context, options) {
            Ok(request) => request,
            Err(error) => {
                fail_stream(
                    &stream,
                    model,
                    &format!("{}: {}", error.code(), error.message()),
                );
                return stream;
            }
        };
        let core = Arc::clone(self);
        let provider = provider.to_string();
        let model = model.clone();
        let output = stream.clone();
        tokio::spawn(async move {
            let call = core.next_call();
            let mut scope = SignalScope::new(call, core.abort_tx.clone());
            let frame = Frame::CallProvider {
                call,
                provider: provider.clone(),
                request,
            };
            let forward = |value: Value| -> Result<(), HostCallError> {
                let event: AssistantMessageEvent =
                    serde_json::from_value(value).map_err(|error| {
                        HostCallError::new(
                            "extension_protocol_error",
                            format!("the provider stream event is malformed: {error}"),
                        )
                    })?;
                output.push(event);
                Ok(())
            };
            let sink = CallSink {
                update: None,
                stream: Some(&forward),
            };
            let context = core.context.clone();
            let outcome = core
                .run_call(call, frame, &context, &mut scope, sink, "provider")
                .await;
            match outcome {
                Ok(CallOutcome::StreamEnd) => output.end(None),
                Ok(CallOutcome::Result(_)) => {
                    let error = HostCallError::new(
                        "extension_protocol_error",
                        "the child answered a provider call without ending its stream",
                    );
                    core.report("provider", &error, None);
                    fail_stream(
                        &output,
                        &model,
                        &format!("{}: {}", error.code(), error.message()),
                    );
                }
                Err(error) => {
                    core.report("provider", &error, error.stack().map(str::to_string));
                    fail_stream(
                        &output,
                        &model,
                        &format!("{}: {}", error.code(), error.message()),
                    );
                }
            }
        });
        stream
    }
}

/// The serializable half of a `stream_simple` request.
fn provider_request(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> Result<Value, HostCallError> {
    let model = serde_json::to_value(model).map_err(|error| {
        HostCallError::new(
            "extension_protocol_error",
            format!("the request model is not JSON: {error}"),
        )
    })?;
    let context = serde_json::to_value(context).map_err(|error| {
        HostCallError::new(
            "extension_protocol_error",
            format!("the request context is not JSON: {error}"),
        )
    })?;
    let options = options.map_or(Value::Null, |options| {
        json!({
            "apiKey": options.api_key,
            "env": options.env,
            "headers": options.headers,
            "timeoutMs": options.timeout_ms,
            "maxRetries": options.max_retries,
            "maxRetryDelayMs": options.max_retry_delay_ms,
            "temperature": options.temperature,
            "samplingParams": options.sampling_params,
            "maxTokens": options.max_tokens,
            "transport": options.transport,
            "cacheRetention": options.cache_retention,
            "sessionId": options.session_id,
            "websocketConnectTimeoutMs": options.websocket_connect_timeout_ms,
            "metadata": options.metadata,
            "toolChoice": options.tool_choice,
            "reasoning": options.reasoning,
            "deferred": options.deferred,
            "thinkingBudgets": options.thinking_budgets,
        })
    });
    Ok(json!({
        "model": model,
        "context": context,
        "options": options,
    }))
}

/// Push the terminal error event and settle the stream.
fn fail_stream(stream: &AssistantMessageEventStream, model: &Model, message: &str) {
    let error = assistant_error(model, message);
    stream.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: error.clone(),
    });
    stream.end(Some(&error));
}

fn assistant_error(model: &Model, message: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: ProviderId(model.provider.0.clone()),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(message.to_string()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}
