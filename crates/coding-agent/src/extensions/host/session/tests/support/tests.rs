//! Shared fixtures for the session tests.

use std::sync::Arc;

use pi_agent_core::types::AgentToolResult;
use pi_ai::types::{Api, AssistantMessage, Model, ProviderId, StopReason, Usage};
use pi_ai::utils::event_stream::AssistantMessageEventStream;
use serde_json::{Value, json};
use tokio::task::JoinHandle;

use super::*;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::ContextActions;
use crate::extensions::types::ToolDefinition;

pub(super) fn demo_tool() -> Frame {
    Frame::RegisterTool {
        tool: json!({
            "name": "demo_tool",
            "label": "Demo",
            "description": "Runs the demo",
            "parameters": {"type": "object"},
        }),
    }
}

pub(super) fn tool_result(details: Value) -> AgentToolResult {
    AgentToolResult {
        content: Vec::new(),
        details,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

pub(super) fn demo_provider() -> Frame {
    Frame::RegisterProvider {
        provider: json!({
            "name": "demo-provider",
            "config": {"name": "Demo", "api": "openai-completions"},
        }),
    }
}

pub(super) fn demo_model() -> Model {
    Model {
        id: "demo-model".to_string(),
        name: "Demo".to_string(),
        api: Api("openai-completions".to_string()),
        provider: ProviderId("demo-provider".to_string()),
        base_url: "http://localhost".to_string(),
        reasoning: false,
        thinking_level_map: None,
        input: Vec::new(),
        cost: pi_ai::types::ModelCost::default(),
        context_window: 8_192,
        max_tokens: 1_024,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

pub(super) fn assistant_message(model: &Model) -> AssistantMessage {
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
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 0,
    }
}

pub(super) async fn connect_options(
    actions: ContextActions,
    ui: Option<Arc<dyn UiCallHandler>>,
) -> HostConnectOptions {
    HostConnectOptions {
        path: "/ext/demo".to_string(),
        resolved_path: "/ext/demo".to_string(),
        source_info: source_info(),
        context: test_context(actions).await,
        ui,
        error_listener: None,
    }
}

pub(super) async fn start_event(
    session: &HostSession<DuplexStream>,
    child: &mut Link<DuplexStream>,
    event: ExtensionEvent,
    actions: ContextActions,
) -> (u64, JoinHandle<HandlerResult>) {
    let handler = handler(session, event.channel());
    let context = test_context(actions).await;
    let task = tokio::spawn(async move { handler(event, context).await });
    let Some(Frame::CallEvent { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    (call, task)
}

pub(super) fn demo_definition(session: &HostSession<DuplexStream>) -> ToolDefinition {
    session
        .extension()
        .tools
        .lock()
        .expect("tools")
        .first()
        .expect("one tool")
        .definition
        .clone()
}

pub(super) async fn start_provider(
    session: &mut HostSession<DuplexStream>,
    child: &mut Link<DuplexStream>,
) -> (AssistantMessageEventStream, Model, u64) {
    let registration = session
        .take_provider_registrations()
        .pop()
        .expect("one provider");
    let stream_simple = Arc::clone(
        registration
            .config
            .stream_simple
            .as_ref()
            .expect("the proxy is bound"),
    );
    let model = demo_model();
    let stream = stream_simple(&model, &pi_ai::types::Context::default(), None);
    let Some(Frame::CallProvider {
        call,
        provider,
        request,
    }) = child.read_frame().await.expect("call")
    else {
        panic!("the host sends a call provider");
    };
    assert_eq!(provider, "demo-provider");
    assert_eq!(request["model"]["id"], json!("demo-model"));
    (stream, model, call)
}
