//! `test/suite/regressions/5998-blocked-tool-terminate.test.ts` at pin
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

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::AgentTool;
use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSessionEvent;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::results::ToolCallResult;
use serde_json::json;

use common::agent_session::SessionSetup;
use common::agent_session::authorize;
use common::agent_session::build;
use common::extension_regressions::assistant_texts;
use common::extension_regressions::assistant_with_calls;
use common::extension_regressions::manager;

/// #5998: a `tool_call` handler can block and terminate the run.
#[tokio::test]
async fn blocked_tool_can_terminate_the_run() {
    let root = "/tmp/pi-ext-reg-blocked-terminate";
    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::clone(&calls);
    let stream: StreamFn = Arc::new(move |_model, _context, _options| {
        stream_calls.fetch_add(1, Ordering::SeqCst);
        let message = assistant_with_calls(&[("call-echo", "echo", json!({"text": "hello"}))], "");
        let stream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
        stream
    });
    let extension = InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "tool_call",
                    Arc::new(|_event: ExtensionEvent, _ctx| {
                        Box::pin(async {
                            HandlerResult::ToolCall(ToolCallResult {
                                block: true,
                                reason: Some("Blocked by terminating policy".to_owned()),
                                terminate: true,
                            })
                        })
                    }),
                )
                .expect("tool_call handler");
                Ok(())
            })
        }),
    };
    let echo = AgentTool {
        tool: pi_ai::types::Tool {
            name: "echo".to_owned(),
            description: "Echo text back".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
            }),
            constrained_sampling: None,
        },
        label: "Echo".to_owned(),
        prepare_arguments: None,
        execute: Arc::new(|_id, _args, _signal, _on_update| {
            Box::pin(async { Err(std::io::Error::other("tool should have been blocked").into()) })
        }),
        replay: None,
        execution_mode: None,
    };
    let mut setup = SessionSetup::new(root, manager(root));
    setup.extensions = vec![extension];
    setup.bind_extensions = true;
    setup.base_tools_override = Some(BTreeMap::from([("echo".to_owned(), echo)]));
    setup.stream = Some(stream);
    let session = build(setup).await;
    authorize(&session).await;
    let ends: Arc<Mutex<Vec<pi_agent_core::types::AgentToolResult>>> =
        Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&ends);
    let _unsubscribe = session.subscribe(Arc::new(move |event| {
        if let AgentSessionEvent::ToolExecutionEnd { result, .. } = event {
            sink.lock().expect("lock").push(result);
        }
    }));

    session.prompt("hi".to_owned(), None).await.expect("prompt");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        !assistant_texts(&session)
            .iter()
            .any(|text| text == "should not run")
    );
    assert_eq!(ends.lock().expect("lock")[0].terminate, Some(true));
    assert!(
        session
            .agent()
            .state()
            .messages
            .iter()
            .any(|message| matches!(message, AgentMessage::Standard(Message::ToolResult(result)) if result.is_error))
    );
}
