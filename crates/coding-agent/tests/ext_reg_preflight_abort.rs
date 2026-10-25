//! `test/suite/regressions/8935-parallel-preflight-abort.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::AgentTool;
use pi_ai::types::Message;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use serde_json::json;

use common::agent_session::SessionSetup;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;
use common::extension_regressions::ok_result;
use common::extension_regressions::scripted;
use common::extension_regressions::tool_result_text;

/// #8935: a later preflight abort prevents every prepared execution.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the block restates upstream's one scenario: fixture, hooks, and transcript assertions inline"
)]
async fn parallel_preflight_abort_skips_execution() {
    let root = "/tmp/pi-ext-reg-preflight-abort";
    let executions: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let preflights: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let result_hooks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let preflight_sink = Arc::clone(&preflights);
    let hook_sink = Arc::clone(&result_hooks);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let preflights = Arc::clone(&preflight_sink);
            let result_hooks = Arc::clone(&hook_sink);
            Box::pin(async move {
                api.on(
                    "tool_call",
                    Arc::new(move |event: ExtensionEvent, ctx| {
                        let preflights = Arc::clone(&preflights);
                        if let ExtensionEvent::ToolCall { input, .. } = &event {
                            let value = input
                                .get("value")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default()
                                .to_owned();
                            preflights.lock().expect("lock").push(value.clone());
                            if value == "second" {
                                ctx.abort().expect("abort");
                            }
                        }
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("tool_call handler");
                api.on(
                    "tool_result",
                    Arc::new(move |event: ExtensionEvent, _ctx| {
                        if let ExtensionEvent::ToolResult { tool_call_id, .. } = event {
                            result_hooks.lock().expect("lock").push(tool_call_id);
                        }
                        Box::pin(async { HandlerResult::None })
                    }),
                )
                .expect("tool_result handler");
                Ok(())
            })
        }),
    };
    let executions_sink = Arc::clone(&executions);
    let write = AgentTool {
        tool: pi_ai::types::Tool {
            name: "external_write".to_owned(),
            description: "Perform an external write".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
            }),
            constrained_sampling: None,
        },
        label: "External write".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, args, _signal, _on_update| {
            let value = args
                .get("value")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let executions = Arc::clone(&executions_sink);
            Box::pin(async move {
                executions.lock().expect("lock").push(value.clone());
                Ok(ok_result(&value))
            })
        }),
        replay: None,
        execution_mode: None,
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(BTreeMap::from([("external_write".to_owned(), write)]));
    setup.stream = Some(scripted(vec![assistant_with_calls(
        &[
            ("call-first", "external_write", json!({"value": "first"})),
            ("call-second", "external_write", json!({"value": "second"})),
        ],
        "",
    )]));
    let session = build(setup).await;
    authorize(&session).await;
    let starts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let ends: Arc<Mutex<Vec<(String, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let start_sink = Arc::clone(&starts);
    let end_sink = Arc::clone(&ends);
    let _unsubscribe = session.subscribe(Arc::new(move |event| match event {
        AgentSessionEvent::ToolExecutionStart { tool_call_id, .. } => {
            start_sink.lock().expect("lock").push(tool_call_id);
        }
        AgentSessionEvent::ToolExecutionEnd {
            tool_call_id,
            is_error,
            ..
        } => {
            end_sink
                .lock()
                .expect("lock")
                .push((tool_call_id, is_error));
        }
        _ => {}
    }));

    session
        .prompt("run both writes".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(
        preflights.lock().expect("lock").clone(),
        ["first".to_owned(), "second".to_owned()]
    );
    assert!(executions.lock().expect("lock").is_empty());
    assert!(result_hooks.lock().expect("lock").is_empty());
    let starts = starts.lock().expect("lock").clone();
    let ends = ends.lock().expect("lock").clone();
    assert_eq!(starts.len(), 2);
    assert_eq!(ends.len(), 2);
    let mut end_ids: Vec<String> = ends.iter().map(|(id, _)| id.clone()).collect();
    end_ids.sort();
    let mut start_ids = starts;
    start_ids.sort();
    assert_eq!(end_ids, start_ids);
    assert!(ends.iter().all(|(_, is_error)| *is_error));

    let tool_results: Vec<(String, String)> = session
        .agent()
        .state()
        .messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(Message::ToolResult(result)) => Some((
                result.tool_call_id.clone(),
                tool_result_text(&Message::ToolResult(result.clone())),
            )),
            _ => None,
        })
        .collect();
    let mut result_ids: Vec<String> = tool_results.iter().map(|(id, _)| id.clone()).collect();
    result_ids.sort();
    assert_eq!(result_ids, start_ids);
    assert!(
        tool_results
            .iter()
            .all(|(_, text)| text == "Operation aborted")
    );
}
