//! `test/suite/regressions/6162-extension-active-tools-next-turn.test.ts`,
//! the switch and additive blocks, at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::ApiSlot;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;
use common::extension_regressions::ok_result;
use common::extension_regressions::switching_tools_extension;
use common::extension_regressions::tool_definition;
use serde_json::json;

/// #6162: `setActiveTools` applies before the next provider request.
#[tokio::test]
async fn active_tools_switch_applies_before_the_next_request() {
    let root = "/tmp/pi-ext-reg-active-next";
    let calls: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let stream_calls = Arc::clone(&calls);
    let request = Arc::new(AtomicUsize::new(0));
    let stream: StreamFn = Arc::new(move |_model, context, _options| {
        let mut names: Vec<String> = context
            .tools
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        names.sort();
        stream_calls.lock().expect("lock").push(names);
        let message = if request.fetch_add(1, Ordering::SeqCst) == 0 {
            assistant_with_calls(&[("call-switch", "switch_tools", json!({}))], "")
        } else {
            assistant_message("done", 0)
        };
        let stream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
        stream
    });
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![switching_tools_extension(Arc::new(Mutex::new(None)))];
    setup.bind_extensions = true;
    setup.stream = Some(stream);
    let session = build(setup).await;
    authorize(&session).await;

    session.set_active_tools_by_name(&["switch_tools".to_owned()]);
    assert_eq!(session.active_tool_names(), ["switch_tools".to_owned()]);
    session
        .prompt("start".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(session.active_tool_names(), ["after_switch".to_owned()]);
    assert_eq!(
        calls.lock().expect("lock").clone(),
        vec![
            vec!["switch_tools".to_owned()],
            vec!["after_switch".to_owned()],
        ]
    );
}

/// #6162: additive active-tool changes land on the current tool result.
#[tokio::test]
async fn active_tool_additions_land_on_the_current_tool_result() {
    let root = "/tmp/pi-ext-reg-active-added";
    let added: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&added);
    let request = Arc::new(AtomicUsize::new(0));
    let stream: StreamFn = Arc::new(move |_model, context, _options| {
        let message = if request.fetch_add(1, Ordering::SeqCst) == 0 {
            assistant_with_calls(&[("call-load", "load_more_tools", json!({}))], "")
        } else {
            let names: Vec<String> = context
                .messages
                .iter()
                .filter_map(|message| match message {
                    Message::ToolResult(result) => result.added_tool_names.clone(),
                    _ => None,
                })
                .flatten()
                .collect();
            *captured.lock().expect("lock") = names;
            assistant_message("done", 0)
        };
        let stream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
        stream
    });
    let api_slot: ApiSlot = Arc::new(Mutex::new(None));
    let load_api = Arc::clone(&api_slot);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&load_api);
            let tool_api = Arc::clone(&api);
            Box::pin(async move {
                let mut load_tool = tool_definition("load_more_tools", "Load more tools", None);
                load_tool.execute = Arc::new(move |_id, _params, _signal, _on_update, _ctx| {
                    let api = Arc::clone(&tool_api);
                    Box::pin(async move {
                        let mut names = api.get_active_tools().expect("active tools");
                        names.push("after_load".to_owned());
                        api.set_active_tools(names).expect("tools load");
                        Ok(ok_result("loaded"))
                    })
                });
                api.register_tool(load_tool).expect("load tool");
                api.register_tool(tool_definition(
                    "after_load",
                    "Tool available after loading",
                    None,
                ))
                .expect("after load tool");
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.stream = Some(stream);
    let session = build(setup).await;
    authorize(&session).await;

    session.set_active_tools_by_name(&["load_more_tools".to_owned()]);
    session
        .prompt("start".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(
        session.active_tool_names(),
        ["load_more_tools".to_owned(), "after_load".to_owned()]
    );
    assert_eq!(
        added.lock().expect("lock").clone(),
        ["after_load".to_owned()]
    );
}
