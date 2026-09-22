//! The lazy stream seam, ported from `packages/ai/src/api/lazy.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use crate::auth::resolve::now_ms;
use crate::types::{
    AssistantMessage, AssistantMessageEvent, BoxedFuture, Model, StopReason, Usage, UsageCost,
};
use crate::utils::event_stream::AssistantMessageEventStream;

/// The failure a lazy-stream setup reports; its display text becomes the
/// stream error's `errorMessage`.
pub type LazyStreamError = Box<dyn std::error::Error + Send + Sync>;

/// The setup failure message, upstream's `createSetupErrorMessage`.
#[must_use]
pub fn setup_error_message(
    model: &Model,
    error: &(dyn std::error::Error + 'static),
) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            reasoning: None,
            total_tokens: 0,
            cost: UsageCost::default(),
        },
        stop_reason: StopReason::Error,
        deferred: None,
        error_message: Some(error.to_string()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: now_ms(),
    }
}

/// Returns a stream synchronously while running async setup (auth resolution,
/// lazy module loading) behind it, upstream's `lazyStream`. Setup failures
/// terminate the stream with an error event.
///
/// Porting restatement: the setup runs on a spawned task, so the returned
/// stream is live immediately like the upstream promise chain.
#[must_use]
pub fn lazy_stream(
    model: &Model,
    setup: impl FnOnce() -> BoxedFuture<'static, Result<AssistantMessageEventStream, LazyStreamError>>
    + Send
    + 'static,
) -> AssistantMessageEventStream {
    let outer = crate::utils::event_stream::assistant_message_event_stream();
    let forward_target = outer.clone();
    let model = model.clone();
    let setup = setup();
    tokio::spawn(async move {
        match setup.await {
            Ok(inner) => {
                let target = forward_target.clone();
                tokio::spawn(async move {
                    while let Some(event) = inner.next().await {
                        target.push(event);
                    }
                    let result = inner.result().await;
                    target.end(Some(&result));
                });
            }
            Err(error) => {
                let static_error: &(dyn std::error::Error + 'static) = error.as_ref();
                let message = setup_error_message(&model, static_error);
                forward_target.push(AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: message.clone(),
                });
                forward_target.end(Some(&message));
            }
        }
    });
    outer
}
