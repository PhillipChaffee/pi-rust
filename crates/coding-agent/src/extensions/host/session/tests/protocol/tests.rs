//! Protocol failures during dispatch and link closure.

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use super::*;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::ContextActions;

use support::demo_provider;

#[tokio::test]
async fn an_unexpected_frame_kills_the_session() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let first = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { first(ExtensionEvent::AgentStart, context).await });
    let _ = child.read_frame().await.expect("call");
    child.write_frame(&Frame::Ready).await.expect("stray frame");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
    assert!(session.is_dead());
    let captured = errors.lock().expect("errors").clone();
    assert!(captured[0].error.contains("extension_protocol_error"));
}

#[tokio::test]
async fn a_stream_end_outside_a_provider_call_reports() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let event = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { event(ExtensionEvent::AgentStart, context).await });
    let Some(Frame::CallEvent { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    child
        .write_frame(&Frame::StreamEnd { call })
        .await
        .expect("stream end");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
    let captured = errors.lock().expect("errors").clone();
    assert!(captured[0].error.contains("outside a provider call"));
}

#[tokio::test]
async fn a_malformed_event_result_reports() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let event = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { event(ExtensionEvent::AgentStart, context).await });
    let Some(Frame::CallEvent { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    child
        .write_frame(&Frame::CallResult {
            call,
            result: json!(42),
        })
        .await
        .expect("result");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
    let captured = errors.lock().expect("errors").clone();
    assert!(captured[0].error.contains("extension payload"));
}

#[tokio::test]
async fn a_provider_result_without_a_stream_end_fails_the_stream() {
    let (mut session, mut child, _errors) =
        ready_session_with(vec![demo_provider()], ContextActions::default(), None).await;
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
    let model = support::demo_model();
    let stream = stream_simple(&model, &pi_ai::types::Context::default(), None);
    let Some(Frame::CallProvider { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call provider");
    };
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    assert!(matches!(
        stream.next().await,
        Some(pi_ai::types::AssistantMessageEvent::Error { .. })
    ));
    assert!(
        stream
            .result()
            .await
            .error_message
            .is_some_and(|message| message.contains("without ending"))
    );
}

#[tokio::test]
async fn close_ends_the_child_stream() {
    let (session, mut child, _errors) =
        ready_session_with(Vec::new(), ContextActions::default(), None).await;
    session.close().await.expect("close");
    assert_eq!(child.read_frame().await.expect("end"), None);
}

#[tokio::test]
async fn a_malformed_frame_fails_the_registration_pass() {
    let options = HostConnectOptions {
        path: "/ext/demo".to_string(),
        resolved_path: "/ext/demo".to_string(),
        source_info: source_info(),
        context: test_context(ContextActions::default()).await,
        ui: None,
        error_listener: None,
    };
    let (host_io, mut child_io) = tokio::io::duplex(1 << 20);
    let task = tokio::spawn(async move { HostSession::connect(host_io, options).await });
    let mut link = Link::new(&mut child_io).expect("temporary child link");
    link.write_frame(&hello()).await.expect("hello");
    let _ = link.read_frame().await.expect("welcome");
    drop(link);
    child_io
        .write_all(&[0, 0, 0, 1, 0xff])
        .await
        .expect("malformed frame");
    let error = task.await.expect("join").expect_err("the connect fails");
    assert!(!error.message().is_empty());
}

#[tokio::test]
async fn an_update_outside_a_tool_call_is_a_protocol_failure() {
    let (session, mut child, _errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let event = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { event(ExtensionEvent::AgentStart, context).await });
    let Some(Frame::CallEvent { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    child
        .write_frame(&Frame::CallUpdate {
            call,
            update: json!({}),
        })
        .await
        .expect("update");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
    assert!(session.is_dead());
}
