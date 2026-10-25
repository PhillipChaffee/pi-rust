//! Command dispatch proxies.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::extensions::runtime::ContextActions;

#[tokio::test]
async fn command_dispatch_runs_and_reports_failures() {
    let (session, mut child, errors) = ready_session_with(
        vec![Frame::RegisterCommand {
            command: json!({"name": "demo"}),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let command = session
        .extension()
        .commands
        .lock()
        .expect("commands")
        .first()
        .expect("one command")
        .clone();
    let run = Arc::clone(&command.handler);
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { run("--flag".to_string(), context).await });
    let Some(Frame::CallCommand { call, name, args }) = child.read_frame().await.expect("call")
    else {
        panic!("the host sends a call command");
    };
    assert_eq!((name.as_str(), args.as_str()), ("demo", "--flag"));
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    task.await.expect("join");

    let run = Arc::clone(&command.handler);
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { run("--again".to_string(), context).await });
    let Some(Frame::CallCommand { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call command");
    };
    child
        .write_frame(&Frame::CallError {
            call,
            code: "command_failed".to_string(),
            message: "boom".to_string(),
            stack: None,
        })
        .await
        .expect("failure");
    task.await.expect("join");
    assert_eq!(errors.lock().expect("errors")[0].event, "command");
}

#[tokio::test]
async fn a_dead_session_skips_commands() {
    let (session, child, _errors) = ready_session_with(
        vec![Frame::RegisterCommand {
            command: json!({"name": "demo"}),
        }],
        ContextActions::default(),
        None,
    )
    .await;
    let command = session
        .extension()
        .commands
        .lock()
        .expect("commands")
        .first()
        .expect("one command")
        .clone();
    let run = Arc::clone(&command.handler);
    let context = test_context(ContextActions::default()).await;
    let task = tokio::spawn(async move { run("--flag".to_string(), context).await });
    drop(child);
    task.await.expect("join");
}
