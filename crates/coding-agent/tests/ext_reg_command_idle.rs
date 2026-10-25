//! `test/suite/regressions/6363-agent-settled-event.test.ts`, the
//! `waitForIdle` block, at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::types::AgentTool;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::core::agent_session::ExtensionBindings;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use serde_json::json;
use tokio::sync::Notify;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::command_actions;
use common::extension_regressions::manager;
use common::extension_regressions::ok_result;
use common::extension_regressions::scripted;

/// #6363: a command's `waitForIdle` waits for session-level settlement.
#[tokio::test]
async fn command_wait_for_idle_waits_for_session_settlement() {
    let root = "/tmp/pi-ext-reg-command-idle";
    let release = Arc::new(Notify::new());
    let release_tool = Arc::clone(&release);
    let wait_tool = AgentTool {
        tool: pi_ai::types::Tool {
            name: "wait".to_owned(),
            description: "Wait until released".to_owned(),
            parameters: json!({ "type": "object", "properties": {} }),
            constrained_sampling: None,
        },
        label: "Wait".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, _args, _signal, _on_update| {
            let release = Arc::clone(&release_tool);
            Box::pin(async move {
                release.notified().await;
                Ok(ok_result("released"))
            })
        }),
        replay: None,
        execution_mode: None,
    };
    let command_results: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let results = Arc::clone(&command_results);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let results = Arc::clone(&results);
            Box::pin(async move {
                api.register_command(
                    "after-idle".to_owned(),
                    Some("Wait for idle".to_owned()),
                    Arc::new(move |_args, ctx| {
                        let results = Arc::clone(&results);
                        Box::pin(async move {
                            ctx.wait_for_idle().await.expect("idle");
                            results
                                .lock()
                                .expect("lock")
                                .push(ctx.is_idle().expect("idle read"));
                        })
                    }),
                )
                .expect("command registers");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.base_tools_override = Some(BTreeMap::from([("wait".to_owned(), wait_tool)]));
    setup.stream = Some(scripted(vec![
        assistant_with_calls(&[("call-wait", "wait", json!({}))], ""),
        assistant_message("done", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;
    session
        .bind_extensions(ExtensionBindings {
            command_context_actions: Some(command_actions(&session)),
            ..ExtensionBindings::default()
        })
        .await
        .expect("bind");

    let tool_started = Arc::new(Notify::new());
    let start_signal = Arc::clone(&tool_started);
    let settled = Arc::new(AtomicUsize::new(0));
    let settled_sink = Arc::clone(&settled);
    let _unsubscribe = session.subscribe(Arc::new(move |event| match event {
        AgentSessionEvent::ToolExecutionStart { tool_name, .. } if tool_name == "wait" => {
            start_signal.notify_one();
        }
        AgentSessionEvent::AgentSettled => {
            settled_sink.fetch_add(1, Ordering::SeqCst);
        }
        _ => {}
    }));

    let prompt_session = Arc::clone(&session);
    let prompt = tokio::spawn(async move { prompt_session.prompt("start".to_owned(), None).await });
    tool_started.notified().await;
    let command_session = Arc::clone(&session);
    let command =
        tokio::spawn(async move { command_session.prompt("/after-idle".to_owned(), None).await });
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    assert!(!command.is_finished());

    release.notify_one();
    prompt.await.expect("join").expect("prompt");
    command.await.expect("join").expect("command");

    assert_eq!(command_results.lock().expect("lock").clone(), vec![true]);
    assert_eq!(settled.load(Ordering::SeqCst), 1);
}
