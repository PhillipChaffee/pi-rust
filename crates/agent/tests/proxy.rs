//! The `streamProxy` suite, ported 1:1 from upstream `test/proxy.test.ts`
//! at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! `vi.stubGlobal("fetch")` restates as the pi-ai mock client mounted on the
//! options' `http_client` seam (`common::proxy_options_with_body`); the wire
//! bodies are built from the [`ProxyAssistantMessageEvent`] values the test
//! scripts, exactly the `data: ${JSON.stringify(event)}` strings upstream
//! sends.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
#![expect(clippy::panic, reason = "tests assert by panicking")]

mod common;

use common::proxy_data_lines;
use common::proxy_done;
use common::proxy_event_type;
use common::proxy_options_with_body;
use common::proxy_toolcall_end;
use common::run_proxy;
use pi_agent_core::ProxyAssistantMessageEvent;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::StopReason;
use serde_json::json;

#[tokio::test]
async fn preserves_tool_call_metadata_received_only_on_toolcall_end() {
    let proxy_events = vec![
        ProxyAssistantMessageEvent::Start,
        ProxyAssistantMessageEvent::ToolcallStart {
            content_index: 0,
            id: "call_test|fc_test".to_owned(),
            tool_name: "lookup".to_owned(),
        },
        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index: 0,
            delta: r#"{"value":"hello"}"#.to_owned(),
        },
        proxy_toolcall_end(
            0,
            json!({
                "type": "toolCall",
                "id": "call_test|fc_test",
                "name": "lookup",
                "arguments": { "value": "hello" },
                "namespace": "dynamic_tools",
            }),
        ),
        proxy_done(StopReason::ToolUse, None),
    ];
    let options = proxy_options_with_body(proxy_data_lines(&proxy_events));

    let (events, result) = run_proxy(options).await;

    let end_tool_call = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call.clone()),
            _ => None,
        })
        .expect("a toolcall_end event");
    assert_eq!(end_tool_call.namespace.as_deref(), Some("dynamic_tools"));
    let AssistantBlock::ToolCall(result_call) = &result.content[0] else {
        panic!("expected a toolCall block at content[0]");
    };
    assert_eq!(
        result_call
            .arguments
            .get("value")
            .and_then(serde_json::Value::as_str),
        Some("hello")
    );
    assert_eq!(result_call.namespace.as_deref(), Some("dynamic_tools"));
}

// Regression tests for https://github.com/earendil-works/pi/issues/8996
#[tokio::test]
async fn processes_terminal_metadata_when_the_event_is_not_newline_terminated() {
    let start = format!(
        "data: {}\n\n",
        serde_json::to_string(&ProxyAssistantMessageEvent::Start)
            .expect("a serializable proxy event")
    );
    let done = format!(
        "data: {}",
        serde_json::to_string(&proxy_done(StopReason::Stop, Some("high")))
            .expect("a serializable proxy event")
    );
    let options = proxy_options_with_body(start + &done);

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["start", "done"]);
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.provider_thinking_level.as_deref(), Some("high"));
}

#[tokio::test]
async fn emits_an_error_instead_of_hanging_when_the_stream_ends_without_a_terminal_event() {
    let options = proxy_options_with_body(format!(
        "data: {}\n\n",
        serde_json::to_string(&ProxyAssistantMessageEvent::Start)
            .expect("a serializable proxy event")
    ));

    let (events, result) = run_proxy(options).await;

    let types: Vec<String> = events.iter().map(proxy_event_type).collect();
    assert_eq!(types, ["start", "error"]);
    assert_eq!(result.stop_reason, StopReason::Error);
    let error_message = result.error_message.expect("an error message");
    assert!(error_message.contains("Connection closed by proxy server"));
}
