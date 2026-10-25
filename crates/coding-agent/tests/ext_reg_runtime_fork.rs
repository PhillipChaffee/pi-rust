//! `test/suite/regressions/8724-in-memory-fork-active-tool.test.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentTool;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
use pi_coding_agent::core::agent_session_runtime::ForkOptions;
use pi_coding_agent::core::agent_session_runtime::create_agent_session_runtime;
use serde_json::json;
use tokio::sync::Notify;

use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::runtime_factory;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;
use common::extension_regressions::ok_result;

/// #8724: an in-memory fork during an active tool turn drops the aborted
/// turn from the replacement session.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the block wires the runtime, the blocking tool, and both transcripts inline like upstream's one scenario"
)]
async fn in_memory_fork_during_active_tool_turn_drops_the_aborted_turn() {
    let fixture = common::sdk_fixture::Fixture::new();
    let cwd = fixture.cwd.clone();
    let agent_dir = fixture.agent_dir.clone();
    let registry = common::sdk_fixture::registry_with_faux().await;
    let model = common::sdk_fixture::faux_model();
    let factory = runtime_factory(common::agent_session::RuntimeFactoryOptions {
        runtime: registry.runtime().clone(),
        model,
        extensions: Arc::new(Vec::new),
        no_extensions: true,
        tools: None,
    });
    let mut runtime = create_agent_session_runtime(
        factory,
        CreateAgentSessionRuntimeOptions {
            cwd: cwd.clone(),
            agent_dir,
            session_manager: manager(&cwd),
            session_start_event: None,
            project_trust_context: None,
        },
    )
    .await
    .expect("runtime");
    authorize(runtime.session()).await;

    let tool_started = Arc::new(Notify::new());
    let started = Arc::clone(&tool_started);
    let block = AgentTool {
        tool: pi_ai::types::Tool {
            name: "block".to_owned(),
            description: "Wait until aborted".to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            constrained_sampling: None,
        },
        label: "Block".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, _args, signal, _on_update| {
            let started = Arc::clone(&started);
            Box::pin(async move {
                started.notify_one();
                if let Some(signal) = signal {
                    signal.cancelled().await;
                } else {
                    std::future::pending::<()>().await;
                }
                Ok(ok_result("tool aborted"))
            })
        }),
        replay: None,
        execution_mode: None,
    };
    runtime.session().agent().set_tools(&[block]);
    runtime
        .session()
        .agent()
        .set_stream_function(Some(common::extension_regressions::scripted(vec![
            assistant_message("first response", 0),
            assistant_with_calls(&[("call-block", "block", json!({}))], ""),
            assistant_message("unused after abort", 0),
        ])));

    runtime
        .session()
        .prompt("first prompt".to_owned(), None)
        .await
        .expect("first prompt");
    let entry_id = runtime.session().user_messages_for_forking()[0].0.clone();

    let outgoing_session = Arc::clone(runtime.session());
    let outgoing = tokio::spawn(async move {
        outgoing_session
            .prompt("start blocking tool".to_owned(), None)
            .await
    });
    tool_started.notified().await;
    let fork = runtime
        .fork(&entry_id, ForkOptions::default())
        .await
        .expect("fork");
    outgoing.await.expect("join").expect("outgoing prompt");
    runtime
        .session()
        .bind_extensions(ExtensionBindings::default())
        .await
        .expect("bind");

    assert!(!fork.cancelled);
    assert_eq!(fork.selected_text.as_deref(), Some("first prompt"));
    assert!(runtime.session().agent().state().messages.is_empty());
    assert!(runtime.session().user_messages_for_forking().is_empty());

    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&captured);
    runtime.session().agent().set_stream_function(Some(Arc::new(
        move |_model, context, _options| {
            *sink.lock().expect("lock") = context
                .messages
                .iter()
                .map(|message| match message {
                    Message::User(_) => "user".to_owned(),
                    Message::Assistant(_) => "assistant".to_owned(),
                    Message::ToolResult(_) => "toolResult".to_owned(),
                })
                .collect();
            let message = assistant_message("next response", 0);
            let stream = assistant_message_event_stream();
            stream.push(AssistantMessageEvent::Done {
                reason: message.stop_reason,
                message,
            });
            stream
        },
    )));
    authorize(runtime.session()).await;
    runtime
        .session()
        .prompt("next prompt".to_owned(), None)
        .await
        .expect("next prompt");

    assert_eq!(captured.lock().expect("lock").clone(), ["user".to_owned()]);
}
