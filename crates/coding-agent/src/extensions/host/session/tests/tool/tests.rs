//! Tool execution proxies.

use std::sync::{Arc, Mutex};

use pi_agent_core::harness::types::AgentHarnessToolUpdateOptions;
use pi_agent_core::types::AgentToolResult;
use serde_json::json;

use super::*;
use crate::extensions::runtime::ContextActions;

use support::{demo_tool, tool_result};

#[tokio::test]
async fn tool_execution_forwards_updates_and_returns_the_result() {
    let (session, mut child, _errors) =
        ready_session_with(vec![demo_tool()], ContextActions::default(), None).await;
    let definition = support::demo_definition(&session);
    assert_eq!(definition.name, "demo_tool");
    let execute = Arc::clone(&definition.execute);
    let updates = Arc::new(Mutex::new(Vec::new()));
    let collected = Arc::clone(&updates);
    let update = move |result: &AgentToolResult,
                       _options: Option<AgentHarnessToolUpdateOptions>| {
        collected.lock().expect("updates").push(result.clone());
    };
    let task = tokio::spawn(async move {
        let tool_context = crate::extensions::CwdContext {
            cwd: "/tool".to_string(),
        };
        execute(
            "call_1",
            &json!({"path": "README.md"}),
            None,
            Some(&update),
            Some(&tool_context),
        )
        .await
    });
    let Some(Frame::CallTool {
        call,
        tool,
        tool_call_id,
        params,
        context,
    }) = child.read_frame().await.expect("call")
    else {
        panic!("the host sends a call tool");
    };
    assert_eq!(tool, "demo_tool");
    assert_eq!(tool_call_id, "call_1");
    assert_eq!(params, json!({"path": "README.md"}));
    assert_eq!(context["cwd"], json!("/tool"));
    child
        .write_frame(&Frame::CallUpdate {
            call,
            update: serde_json::to_value(tool_result(json!({"step": 1}))).expect("update"),
        })
        .await
        .expect("update");
    child
        .write_frame(&Frame::CallResult {
            call,
            result: serde_json::to_value(tool_result(json!({"done": true}))).expect("result"),
        })
        .await
        .expect("result");
    let result = task.await.expect("join").expect("the tool succeeds");
    assert_eq!(result.details, json!({"done": true}));
    assert_eq!(updates.lock().expect("updates").len(), 1);
}

#[tokio::test]
async fn tool_failure_reads_as_an_error() {
    let (session, mut child, errors) =
        ready_session_with(vec![demo_tool()], ContextActions::default(), None).await;
    let definition = support::demo_definition(&session);
    let execute = Arc::clone(&definition.execute);
    let task = tokio::spawn(async move { execute("call_2", &json!({}), None, None, None).await });
    let Some(Frame::CallTool { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call tool");
    };
    child
        .write_frame(&Frame::CallError {
            call,
            code: "tool_failed".to_string(),
            message: "nope".to_string(),
            stack: None,
        })
        .await
        .expect("failure");
    let error = task.await.expect("join").expect_err("the tool fails");
    assert!(error.to_string().contains("nope"));
    assert_eq!(errors.lock().expect("errors")[0].event, "tool");
}

#[tokio::test]
async fn malformed_tool_updates_and_results_fail_the_call() {
    let (session, mut child, errors) =
        ready_session_with(vec![demo_tool()], ContextActions::default(), None).await;
    let definition = support::demo_definition(&session);
    let execute = Arc::clone(&definition.execute);
    let task = tokio::spawn(async move { execute("call_3", &json!({}), None, None, None).await });
    let Some(Frame::CallTool { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call tool");
    };
    child
        .write_frame(&Frame::CallUpdate {
            call,
            update: json!(5),
        })
        .await
        .expect("update");
    let error = task
        .await
        .expect("join")
        .expect_err("the update is malformed");
    assert!(error.to_string().contains("tool result"));

    let execute = Arc::clone(&definition.execute);
    let task = tokio::spawn(async move { execute("call_4", &json!({}), None, None, None).await });
    let Some(Frame::CallTool { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call tool");
    };
    child
        .write_frame(&Frame::CallResult {
            call,
            result: json!(5),
        })
        .await
        .expect("result");
    let error = task
        .await
        .expect("join")
        .expect_err("the result is malformed");
    assert!(error.to_string().contains("tool result"));
    assert!(errors.lock().expect("errors").len() >= 2);
}

#[tokio::test]
async fn a_dead_session_fails_tool_calls_without_writing() {
    let (session, child, _errors) =
        ready_session_with(vec![demo_tool()], ContextActions::default(), None).await;
    let definition = support::demo_definition(&session);
    let execute = Arc::clone(&definition.execute);
    let first = tokio::spawn(async move { execute("call_5", &json!({}), None, None, None).await });
    drop(child);
    let error = first.await.expect("join").expect_err("the child is gone");
    assert!(!error.to_string().is_empty());
    let execute = Arc::clone(&definition.execute);
    let error = execute("call_6", &json!({}), None, None, None)
        .await
        .expect_err("the session is dead");
    assert!(error.to_string().contains("exited"));
}
