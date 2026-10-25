//! Signal ids and cancellation pushes.

use std::sync::Arc;

use pi_agent_core::harness::context::{background_context, with_cancel};
use serde_json::{Value, json};

use super::*;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::runtime::ContextActions;

#[tokio::test]
async fn signal_cancellation_pushes_a_signal_frame() {
    let (_event_context, event_controller) = with_cancel(&background_context());
    let signal = event_controller.signal().clone();
    let (_read_context, controller) = with_cancel(&background_context());
    let read_signal = controller.signal().clone();
    let actions = ContextActions {
        get_signal: Some(Arc::new(move || Some(read_signal.clone()))),
        ..Default::default()
    };
    let handler_actions = actions.clone();
    let (session, mut child, _errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "before_agent_start".to_string(),
        }],
        actions,
        None,
    )
    .await;
    let handler = handler(&session, "before_agent_start");
    let context = test_context(handler_actions).await;
    let task = tokio::spawn(async move {
        handler(
            ExtensionEvent::BeforeAgentStart {
                prompt: "hello".to_string(),
                images: None,
                system_prompt: "system".to_string(),
                system_prompt_options: crate::system_prompt::BuildSystemPromptOptions {
                    cwd: "/tmp/project".to_string(),
                    ..Default::default()
                },
                signal: Some(signal),
            },
            context,
        )
        .await
    });
    let Some(Frame::CallEvent { call, payload, .. }) = child.read_frame().await.expect("call")
    else {
        panic!("the host sends a call event");
    };
    assert_eq!(payload["signal"], json!(1));
    child
        .write_frame(&Frame::ContextCall {
            call,
            id: 1,
            method: "get_signal".to_string(),
            params: None,
        })
        .await
        .expect("context call");
    let Some(Frame::ContextResult { value, .. }) = child.read_frame().await.expect("answer") else {
        panic!("the read answers");
    };
    let signal_id = value["signal"].as_u64().expect("a signal id");
    controller.abort("stop");
    match child.read_frame().await.expect("signal frame") {
        Some(Frame::Signal {
            call: owner,
            signal: pushed,
            aborted,
        }) => assert_eq!((owner, pushed, aborted), (call, signal_id, true)),
        other => panic!("the host pushes a signal frame, got {other:?}"),
    }
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    let _ = task.await.expect("join");
}
