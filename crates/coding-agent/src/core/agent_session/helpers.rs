//! Shared message and timing helpers.

/// A custom agent message, upstream's `createCustomMessage` call.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the constructor owns the custom type; upstream carries a string"
)]
pub(super) fn custom_message(
    custom_type: String,
    content: pi_ai::types::UserContent,
    display: bool,
    details: Option<serde_json::Value>,
) -> pi_agent_core::types::AgentMessage {
    let timestamp = crate::session_manager::now_iso8601();
    pi_agent_core::types::AgentMessage::Custom(crate::messages::create_custom_message(
        &custom_type,
        content,
        display,
        details,
        &timestamp,
    ))
}

/// A user message carrying text and optional images.
pub(super) fn user_message(
    text: String,
    images: Option<Vec<pi_ai::types::ImageContent>>,
) -> pi_agent_core::types::AgentMessage {
    let mut content = vec![pi_ai::types::UserBlock::Text(pi_ai::types::TextContent {
        text,
        text_signature: None,
    })];
    if let Some(images) = images {
        content.extend(images.into_iter().map(pi_ai::types::UserBlock::Image));
    }
    pi_agent_core::types::AgentMessage::Standard(pi_ai::types::Message::User(
        pi_ai::types::UserMessage {
            content: pi_ai::types::UserContent::Blocks(content),
            timestamp: now_ms(),
        },
    ))
}

/// The current epoch milliseconds, upstream's `Date.now()`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "millisecond counts since the epoch fit i64 for any system clock this code runs on"
)]
pub(super) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_millis() as i64)
}

/// The prompt preflight observer, upstream's `preflightResult?.()`.
pub(super) fn notify_preflight(preflight: Option<&(dyn Fn(bool) + Send + Sync)>, success: bool) {
    if let Some(preflight) = preflight {
        preflight(success);
    }
}

/// A bridged extension signal for a run's cancellation token, plus the
/// done token that releases the watcher when the operation settles.
pub(super) fn bridge_signal(
    token: Option<&tokio_util::sync::CancellationToken>,
) -> (
    Option<pi_agent_core::harness::context::AbortSignal>,
    Option<tokio_util::sync::CancellationToken>,
) {
    let Some(token) = token else {
        return (None, None);
    };
    let (_context, controller) = pi_agent_core::harness::context::with_cancel(
        &pi_agent_core::harness::context::background_context(),
    );
    let done = tokio_util::sync::CancellationToken::new();
    let signal = controller.signal().clone();
    let token = token.clone();
    let done_for_task = done.clone();
    tokio::spawn(async move {
        tokio::select! {
            () = token.cancelled() => controller.abort_without_reason(),
            () = done_for_task.cancelled() => {}
        }
    });
    (Some(signal), Some(done))
}

/// The per-model settings key upstream's `${provider}/${id}` spells.
pub(super) fn model_key(model: &pi_ai::types::Model) -> crate::settings_manager::ModelKey {
    crate::settings_manager::ModelKey {
        provider: model.provider.0.clone(),
        id: model.id.clone(),
    }
}

/// The retry policy the compaction/branch-summary calls receive.
pub(super) fn retry_policy(
    settings: &crate::settings_manager::RetrySettings,
) -> pi_ai::utils::retry::RetryPolicy {
    pi_ai::utils::retry::RetryPolicy {
        enabled: settings.enabled,
        max_retries: u32::try_from(settings.max_retries.max(0)).unwrap_or(0),
        base_delay_ms: u64::try_from(settings.base_delay_ms.max(0)).unwrap_or(0),
        max_agent_delay_ms: Some(u64::try_from(settings.max_agent_delay_ms.max(0)).unwrap_or(0)),
    }
}
