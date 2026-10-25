//! Handshake failures and event dispatch.

use super::*;

use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::SessionShutdownReason;
use crate::extensions::results::{HandlerResult, SessionBeforeResult};

#[tokio::test]
async fn handshake_refuses_a_wrong_protocol_version() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child
        .write_frame(&Frame::Hello {
            protocol_version: EXTENSION_PROTOCOL_VERSION + 1,
            extension: ExtensionIdentity {
                name: "demo".to_string(),
                version: "0.1.0".to_string(),
            },
        })
        .await
        .expect("hello");
    assert!(matches!(
        child.read_frame().await.expect("refusal"),
        Some(Frame::HelloError { code, .. }) if code == "protocol_version"
    ));
    let error = task.await.expect("join").expect_err("the handshake fails");
    assert!(error.message().contains("wire version"));
}

#[tokio::test]
async fn handshake_refuses_a_non_hello_first_frame() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child.write_frame(&Frame::Ready).await.expect("ready");
    assert!(matches!(
        child.read_frame().await.expect("refusal"),
        Some(Frame::HelloError { code, .. }) if code == "handshake"
    ));
    let error = task.await.expect("join").expect_err("the handshake fails");
    assert!(error.message().contains("first frame"));
}

#[tokio::test]
async fn handshake_fails_when_the_child_closes_first() {
    let (task, child, _errors) = open_session(ContextActions::default(), None).await;
    drop(child);
    let error = task.await.expect("join").expect_err("the handshake fails");
    assert!(error.message().contains("before its hello"));
}

#[tokio::test]
async fn registration_fails_closed_on_an_unknown_channel() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child.write_frame(&hello()).await.expect("hello");
    let _ = child.read_frame().await.expect("welcome");
    child
        .write_frame(&Frame::RegisterEvent {
            channel: "not_a_channel".to_string(),
        })
        .await
        .expect("registration");
    let error = task
        .await
        .expect("join")
        .expect_err("the registration fails");
    assert!(error.message().contains("unknown event channel"));
}

#[tokio::test]
async fn registration_fails_closed_before_ready() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child.write_frame(&hello()).await.expect("hello");
    let _ = child.read_frame().await.expect("welcome");
    child
        .write_frame(&Frame::CallResult {
            call: 1,
            result: serde_json::Value::Null,
        })
        .await
        .expect("unexpected frame");
    let error = task
        .await
        .expect("join")
        .expect_err("the registration fails");
    assert!(error.message().contains("before ready"));
}

#[tokio::test]
async fn event_dispatch_round_trips_a_result() {
    let (session, mut child, _errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "session_before_switch".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let handler = handler(&session, "session_before_switch");
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move {
        handler(
            ExtensionEvent::SessionBeforeSwitch {
                reason: SessionShutdownReason::New,
                target_session_file: None,
            },
            context,
        )
        .await
    });
    let Some(Frame::CallEvent {
        call,
        channel,
        payload,
    }) = child.read_frame().await.expect("call")
    else {
        panic!("the host sends a call event");
    };
    assert_eq!(channel, "session_before_switch");
    assert_eq!(payload["type"], "session_before_switch");
    child
        .write_frame(&Frame::CallResult {
            call,
            result: serde_json::json!({"cancel": true}),
        })
        .await
        .expect("result");
    assert!(matches!(
        task.await.expect("join"),
        HandlerResult::SessionBefore(SessionBeforeResult::Switch { cancel: true })
    ));
}

#[tokio::test]
async fn event_handler_failure_reads_as_no_result_and_reports() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let (call, task) = support::start_event(
        &session,
        &mut child,
        ExtensionEvent::AgentStart,
        ContextActions::default(),
    )
    .await;
    child
        .write_frame(&Frame::CallError {
            call,
            code: "handler_failed".to_string(),
            message: "boom".to_string(),
            stack: Some("at demo".to_string()),
        })
        .await
        .expect("failure");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
    let errors = errors.lock().expect("errors");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].event, "agent_start");
    assert!(errors[0].error.contains("handler_failed"));
    assert_eq!(errors[0].stack.as_deref(), Some("at demo"));
}

#[tokio::test]
async fn registration_requires_ready_before_the_links_ends() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child.write_frame(&hello()).await.expect("hello");
    let _ = child.read_frame().await.expect("welcome");
    drop(child);
    let error = task
        .await
        .expect("join")
        .expect_err("the registration fails");
    assert!(error.message().contains("before it was ready"));
}

#[tokio::test]
async fn registration_rejects_a_malformed_declaration() {
    let (task, mut child, _errors) = open_session(ContextActions::default(), None).await;
    child.write_frame(&hello()).await.expect("hello");
    let _ = child.read_frame().await.expect("welcome");
    child
        .write_frame(&Frame::RegisterTool {
            tool: serde_json::json!({"name": 1}),
        })
        .await
        .expect("registration");
    let error = task
        .await
        .expect("join")
        .expect_err("the registration fails");
    assert!(error.message().contains("register_tool"));
}
