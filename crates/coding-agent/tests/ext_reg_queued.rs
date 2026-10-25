//! `test/suite/regressions/2023-queued-slash-command-followup.test.ts` at
//! pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentTool;
use pi_ai::types::UserContent;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::runner::DeliverAs;
use pi_coding_agent::extensions::runner::SendUserMessageOptions;
use serde_json::json;
use tokio::sync::Notify;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::ApiSlot;
use common::extension_regressions::assistant_texts;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;
use common::extension_regressions::ok_result;
use common::extension_regressions::scripted;
use common::extension_regressions::user_texts;

/// #2023: an extension-origin queued slash-command follow-up is raw user
/// text, never a command dispatch.
#[tokio::test]
async fn queued_slash_command_follow_up_stays_raw_user_text() {
    let root = "/tmp/pi-ext-reg-queued-command";
    std::fs::create_dir_all(root).expect("root");
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let command_runs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let handler_slot = Arc::clone(&api_slot);
    let handler_runs = Arc::clone(&command_runs);
    let command_extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&handler_slot);
            let runs = Arc::clone(&handler_runs);
            Box::pin(async move {
                api.register_command(
                    "testcmd".to_owned(),
                    Some("Test command".to_owned()),
                    Arc::new(move |args, _ctx| {
                        runs.lock().expect("lock").push(args);
                        Box::pin(async {})
                    }),
                )
                .expect("command registers");
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    };
    let release = Arc::new(Notify::new());
    let release_tool = Arc::clone(&release);
    let wait_tool = AgentTool {
        tool: pi_ai::types::Tool {
            name: "wait".to_owned(),
            description: "Wait for the test to release execution".to_owned(),
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
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![command_extension];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(BTreeMap::from([("wait".to_owned(), wait_tool)]));
    setup.stream = Some(scripted(vec![
        assistant_with_calls(&[("call-wait", "wait", json!({}))], ""),
        assistant_message("first turn complete", 0),
        assistant_message("queued follow-up handled by model", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;

    let tool_started = Arc::new(Notify::new());
    let start_signal = Arc::clone(&tool_started);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::ToolExecutionStart { tool_name, .. } = event
            && tool_name == "wait"
        {
            start_signal.notify_one();
        }
    }));

    let prompt_session = Arc::clone(&session);
    let prompt = tokio::spawn(async move { prompt_session.prompt("start".to_owned(), None).await });
    tool_started.notified().await;
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    api_slot
        .lock()
        .expect("lock")
        .as_ref()
        .expect("api")
        .send_user_message(
            UserContent::Text("/testcmd queued".to_owned()),
            Some(SendUserMessageOptions {
                deliver_as: Some(DeliverAs::FollowUp),
                expand_prompt_templates: None,
            }),
        )
        .expect("message sends");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    release.notify_one();

    prompt.await.expect("join").expect("prompt");
    assert!(command_runs.lock().expect("lock").is_empty());
    assert_eq!(
        user_texts(&session),
        ["start".to_owned(), "/testcmd queued".to_owned()]
    );
    assert!(
        assistant_texts(&session)
            .iter()
            .any(|text| text == "queued follow-up handled by model")
    );
}
