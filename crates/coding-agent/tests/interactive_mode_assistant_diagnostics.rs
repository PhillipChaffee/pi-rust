//! `test/interactive-mode-assistant-diagnostics.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Restatements: upstream builds a `Container` and a settings stub for
//! `this`; the port's frame owns the transcript container and settings
//! manager, so the test drives the real `maybe_show_assistant_diagnostics`
//! and reads the rendered chat.

#![expect(
    clippy::expect_used,
    reason = "the fixture message pins the wire shape; a malformed fixture is a test bug"
)]

mod common;

use common::interactive::{Harness, HarnessOptions};
use pi_ai::types::AssistantMessage;
use pi_coding_agent::settings_manager::Settings;

fn message() -> AssistantMessage {
    serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": "survived" }],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "claude-fable-5-1",
        "usage": {
            "input": 1,
            "output": 1,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 2,
            "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
        },
        "stopReason": "stop",
        "timestamp": 1,
        "diagnostics": [{
            "type": "anthropic_input_transformations",
            "timestamp": 1,
            "details": {
                "transformations": [{
                    "type": "thinking_dropped",
                    "path": "messages.2.content.0",
                    "reason": "prefix_binding_mismatch"
                }]
            }
        }]
    }))
    .expect("fixture parses")
}

#[tokio::test]
async fn shows_anthropic_thinking_drops_when_cache_miss_notices_are_enabled() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), true.into());
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.maybe_show_assistant_diagnostics(&message());

    assert!(harness.chat_text().contains(
        "Anthropic dropped thinking block: prefix_binding_mismatch at messages.2.content.0"
    ));
}

#[tokio::test]
async fn stays_silent_when_cache_miss_notices_are_disabled() {
    let mut settings = Settings::new();
    settings.insert("showCacheMissNotices".to_owned(), false.into());
    let harness = Harness::new(HarnessOptions {
        settings,
        ..HarnessOptions::default()
    })
    .await;

    harness.frame.maybe_show_assistant_diagnostics(&message());

    assert_eq!(harness.frame.chat_container().children.borrow().len(), 0);
}
