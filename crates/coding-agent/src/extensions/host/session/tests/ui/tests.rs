//! The UI pass-through.

use std::sync::Arc;

use pi_ai::types::BoxedFuture;
use serde_json::{Value, json};

use super::*;
use crate::extensions::events::ExtensionEvent;

use super::super::super::context::{HostCallError, UiCallHandler};

struct EchoUi;

impl UiCallHandler for EchoUi {
    fn call(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> BoxedFuture<'static, Result<Value, HostCallError>> {
        let method = method.to_string();
        Box::pin(async move { Ok(json!({"method": method, "params": params})) })
    }
}

#[tokio::test]
async fn ui_calls_pass_through_a_bound_handler() {
    let (session, mut child, _errors) = ready_session_with(
        vec![Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        }],
        ContextActions::default(),
        Some(Arc::new(EchoUi)),
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
        .write_frame(&Frame::UiCall {
            call,
            id: 1,
            method: "select".to_string(),
            params: Some(json!({"options": ["a"]})),
        })
        .await
        .expect("ui call");
    assert_eq!(
        child.read_frame().await.expect("answer"),
        Some(Frame::UiResult {
            call,
            id: 1,
            value: json!({"method": "select", "params": {"options": ["a"]}}),
        })
    );
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    let _ = task.await.expect("join");
}

#[tokio::test]
async fn ui_calls_without_a_handler_answer_unsupported() {
    let (session, mut child, _errors) = ready_session_with(
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
        .write_frame(&Frame::UiCall {
            call,
            id: 2,
            method: "select".to_string(),
            params: None,
        })
        .await
        .expect("ui call");
    assert!(matches!(
        child.read_frame().await.expect("answer"),
        Some(Frame::UiError { code, .. }) if code == "unsupported"
    ));
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    let _ = task.await.expect("join");
}
