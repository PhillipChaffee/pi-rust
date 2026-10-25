//! `test/suite/regressions/1717-2113-agent-session-event-settlement.test.ts`
//! and `3982-message-end-cost-override.test.ts` at pin
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
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::MessageEndResult;
use serde_json::json;

use common::agent_session::SessionSetup;
use common::agent_session::assistant_message;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::Session;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::echo_tool;
use common::extension_regressions::manager;
use common::extension_regressions::roles;
use common::extension_regressions::scripted;

/// #1717/#2113: the settled assistant message persists before its slow
/// `message_end` handler, so tool results keep the transcript order.
#[tokio::test]
async fn persisted_message_order_survives_slow_message_end_handlers() {
    let root = "/tmp/pi-ext-reg-event-order";
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![common::agent_session::slow_message_end_extension()];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(BTreeMap::from([("echo".to_owned(), echo_tool())]));
    setup.stream = Some(scripted(vec![
        assistant_with_calls(
            &[
                ("call-1", "echo", json!({"text": "one"})),
                ("call-2", "echo", json!({"text": "two"})),
            ],
            "",
        ),
        assistant_message("done", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;

    session
        .prompt("run tools".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(
        roles(&session),
        ["user", "assistant", "toolResult", "toolResult", "assistant"]
    );
    let first_tool_result = roles(&session)
        .iter()
        .position(|role| role == "toolResult")
        .expect("a tool result");
    assert!(first_tool_result > 0);
    assert_eq!(roles(&session)[first_tool_result - 1], "assistant");
}

/// #1717/#2113: `tool_call` handlers run after the assistant tool-use
/// message is settled in the session.
#[tokio::test]
async fn tool_call_handlers_see_the_settled_assistant_message() {
    let root = "/tmp/pi-ext-reg-event-settled";
    let slot: Arc<Mutex<Option<Session>>> = Arc::new(Mutex::new(None));
    let snapshots: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let handler_slot = Arc::clone(&slot);
    let handler_snapshots = Arc::clone(&snapshots);
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&handler_slot);
            let snapshots = Arc::clone(&handler_snapshots);
            Box::pin(async move {
                api.on(
                    "tool_call",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        let slot = Arc::clone(&slot);
                        let snapshots = Arc::clone(&snapshots);
                        Box::pin(async move {
                            let session = { slot.lock().expect("lock").as_ref().cloned() };
                            if let Some(session) = session {
                                snapshots.lock().expect("lock").push(roles(&session));
                            }
                            HandlerResult::None
                        })
                    }),
                )
                .expect("tool_call handler");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(BTreeMap::from([("echo".to_owned(), echo_tool())]));
    setup.stream = Some(scripted(vec![
        assistant_with_calls(&[("call-1", "echo", json!({"text": "hello"}))], ""),
        assistant_message("done", 0),
    ]));
    let session = build(setup).await;
    authorize(&session).await;
    *slot.lock().expect("lock") = Some(Arc::clone(&session));

    session
        .prompt("run tool".to_owned(), None)
        .await
        .expect("prompt");

    assert_eq!(
        snapshots.lock().expect("lock").clone(),
        vec![vec!["user".to_owned(), "assistant".to_owned()]]
    );
}

/// #3982: a `message_end` replacement can rewrite finalized usage cost.
#[tokio::test]
async fn message_end_can_override_assistant_usage_cost() {
    let root = "/tmp/pi-ext-reg-cost-override";
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "message_end",
                    Arc::new(|event: ExtensionEvent, _ctx| {
                        let ExtensionEvent::MessageEnd { message } = event else {
                            return Box::pin(async { HandlerResult::None });
                        };
                        let AgentMessage::Standard(pi_ai::types::Message::Assistant(
                            mut assistant,
                        )) = message
                        else {
                            return Box::pin(async { HandlerResult::None });
                        };
                        assistant.usage.cost.total = 0.123;
                        Box::pin(async move {
                            HandlerResult::MessageEnd(MessageEndResult {
                                message: Some(AgentMessage::Standard(
                                    pi_ai::types::Message::Assistant(assistant),
                                )),
                            })
                        })
                    }),
                )
                .expect("message_end handler");
                Ok(())
            })
        }),
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.stream = Some(scripted(vec![assistant_message("hello", 0)]));
    let session = build(setup).await;
    authorize(&session).await;
    let events: Arc<Mutex<Vec<AgentMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&events);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::MessageEnd { message } = event {
            sink.lock().expect("lock").push(message);
        }
    }));

    session.prompt("hi".to_owned(), None).await.expect("prompt");

    let assistant = session
        .agent()
        .state()
        .messages
        .iter()
        .find_map(|message| match message {
            AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant)) => {
                Some(assistant.clone())
            }
            _ => None,
        })
        .expect("assistant message");
    assert!((assistant.usage.cost.total - 0.123).abs() < f64::EPSILON);

    let event_assistant = events
        .lock()
        .expect("lock")
        .iter()
        .find_map(|message| match message {
            AgentMessage::Standard(pi_ai::types::Message::Assistant(assistant)) => {
                Some(assistant.clone())
            }
            _ => None,
        })
        .expect("assistant message_end event");
    assert!((event_assistant.usage.cost.total - 0.123).abs() < f64::EPSILON);
}
