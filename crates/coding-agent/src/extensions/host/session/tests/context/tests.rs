//! Context-call answers.

use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::results::HandlerResult;
use crate::extensions::runtime::ContextActions;

use super::super::super::context::{HostCallError, call_context};

#[tokio::test]
async fn context_calls_are_answered_in_scope() {
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
    let cases: Vec<(&str, Option<Value>, Value)> = vec![
        ("cwd", None, json!("/tmp/project")),
        ("mode", None, json!("print")),
        ("has_ui", None, json!(false)),
        ("model", None, Value::Null),
        ("scoped_models", None, json!([])),
        ("thinking_level", None, json!("off")),
        ("is_idle", None, json!(true)),
        ("is_project_trusted", None, json!(true)),
        ("has_pending_messages", None, json!(false)),
        ("get_context_usage", None, Value::Null),
        ("get_system_prompt", None, json!("")),
        (
            "get_system_prompt_options",
            None,
            json!({"cwd": "/tmp/project"}),
        ),
        (
            "compact",
            Some(json!({"customInstructions": "keep"})),
            Value::Null,
        ),
        ("abort", None, Value::Null),
        ("shutdown", None, Value::Null),
        ("get_signal", None, json!({"signal": null})),
        ("session.getSessionDir", None, json!("/tmp/sessions")),
        ("session.getCwd", None, json!("/tmp/project")),
        ("session.getSessionName", None, Value::Null),
        ("session.getSessionFile", None, Value::Null),
    ];
    for (index, (method, params, expected)) in cases.into_iter().enumerate() {
        let id = u64::try_from(index + 1).expect("small id");
        child
            .write_frame(&Frame::ContextCall {
                call,
                id,
                method: method.to_string(),
                params,
            })
            .await
            .expect("context call");
        assert_eq!(
            child.read_frame().await.expect("answer"),
            Some(Frame::ContextResult {
                call,
                id,
                value: expected,
            }),
        );
    }
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    assert!(matches!(task.await.expect("join"), HandlerResult::None));
}

#[tokio::test]
async fn domain_reads_answer_session_and_model_shapes() {
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
    for (id, method, array) in [
        (1_u64, "session.getSessionId", false),
        (2, "models.getAll", true),
        (3, "models.getAvailable", true),
    ] {
        child
            .write_frame(&Frame::ContextCall {
                call,
                id,
                method: method.to_string(),
                params: None,
            })
            .await
            .expect("context call");
        let Some(Frame::ContextResult { value, .. }) = child.read_frame().await.expect("answer")
        else {
            panic!("the domain read answers a value");
        };
        if array {
            assert!(value.is_array(), "{method} answers an array");
        } else {
            assert!(
                value.as_str().is_some_and(|id| !id.is_empty()),
                "{method} answers a non-empty id"
            );
        }
    }
    child
        .write_frame(&Frame::ContextCall {
            call,
            id: 4,
            method: "models.find".to_string(),
            params: Some(json!({"provider": "demo", "model": "missing"})),
        })
        .await
        .expect("context call");
    assert_eq!(
        child.read_frame().await.expect("answer"),
        Some(Frame::ContextResult {
            call,
            id: 4,
            value: Value::Null,
        }),
    );
    child
        .write_frame(&Frame::ContextCall {
            call,
            id: 5,
            method: "models.find".to_string(),
            params: Some(json!({"provider": "demo"})),
        })
        .await
        .expect("context call");
    assert!(matches!(
        child.read_frame().await.expect("answer"),
        Some(Frame::ContextError { code, .. }) if code == "invalid_params"
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

#[tokio::test]
async fn unknown_context_methods_answer_unknown_method() {
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
        .write_frame(&Frame::ContextCall {
            call,
            id: 1,
            method: "not_a_method".to_string(),
            params: None,
        })
        .await
        .expect("context call");
    assert!(matches!(
        child.read_frame().await.expect("answer"),
        Some(Frame::ContextError { code, .. }) if code == "unknown_method"
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

#[tokio::test]
async fn stale_accessors_answer_stale() {
    let context = HandlerContext::new(
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        Arc::new(|| Err("the runtime is stale".to_string())),
        Arc::new(ContextActions::default()),
    );
    let error = call_context(&context, "cwd", None, false, &mut |_signal| 1)
        .expect_err("the accessor is stale");
    assert_eq!(error.code(), "stale");
}

#[test]
fn host_call_errors_carry_codes() {
    let error = HostCallError::unknown_method("nope");
    assert_eq!(error.code(), "unknown_method");
    assert!(error.to_string().contains("nope"));
}

#[tokio::test]
async fn rich_prompt_options_keep_upstream_shapes() {
    let model = support::demo_model();
    let scoped = crate::extensions::reasons::ScopedModel {
        model,
        thinking_level: Some(pi_agent_core::types::ThinkingLevel::High),
    };
    let actions = ContextActions {
        get_scoped_models: Some(Arc::new(move || vec![scoped.clone()])),
        get_system_prompt_options: Some(Arc::new(|| {
            crate::system_prompt::BuildSystemPromptOptions {
                custom_prompt: Some("custom".to_string()),
                selected_tools: Some(vec!["read".to_string()]),
                tool_snippets: Some(std::collections::HashMap::from([(
                    "read".to_string(),
                    "Reads".to_string(),
                )])),
                prompt_guidelines: Some(vec!["be nice".to_string()]),
                append_system_prompt: Some("appended".to_string()),
                cwd: "/tmp/project".to_string(),
                context_files: Some(vec![crate::system_prompt::ContextFile {
                    path: "/tmp/AGENTS.md".to_string(),
                    content: "rules".to_string(),
                }]),
                skills: Some(vec![crate::skills::Skill {
                    name: "demo".to_string(),
                    description: "Demo skill".to_string(),
                    file_path: "/tmp/SKILL.md".to_string(),
                    base_dir: "/tmp".to_string(),
                    source_info: crate::source_info::create_synthetic_source_info(
                        "/tmp/SKILL.md",
                        &SyntheticSourceOptions {
                            source: "auto".to_string(),
                            scope: Some(SourceScope::User),
                            origin: Some(SourceOrigin::TopLevel),
                            base_dir: None,
                        },
                    ),
                    disable_model_invocation: true,
                }]),
            }
        })),
        ..Default::default()
    };
    let context = test_context(actions).await;
    let scoped = call_context(&context, "scoped_models", None, false, &mut |_signal| 1)
        .expect("scoped models");
    assert_eq!(scoped[0]["model"]["id"], json!("demo-model"));
    assert_eq!(scoped[0]["thinkingLevel"], json!("high"));
    let options = call_context(
        &context,
        "get_system_prompt_options",
        None,
        false,
        &mut |_signal| 1,
    )
    .expect("prompt options");
    assert_eq!(options["customPrompt"], json!("custom"));
    assert_eq!(options["selectedTools"], json!(["read"]));
    assert_eq!(options["toolSnippets"]["read"], json!("Reads"));
    assert_eq!(options["promptGuidelines"], json!(["be nice"]));
    assert_eq!(options["appendSystemPrompt"], json!("appended"));
    assert_eq!(options["contextFiles"][0]["path"], json!("/tmp/AGENTS.md"));
    assert_eq!(options["skills"][0]["name"], json!("demo"));
    assert_eq!(options["skills"][0]["sourceInfo"]["scope"], json!("user"));
    assert_eq!(options["skills"][0]["disableModelInvocation"], json!(true));
}

#[tokio::test]
async fn every_mode_reads_its_wire_name() {
    for (mode, name) in [
        (ExtensionMode::Tui, "tui"),
        (ExtensionMode::Rpc, "rpc"),
        (ExtensionMode::Json, "json"),
        (ExtensionMode::Print, "print"),
    ] {
        let context = HandlerContext::new(
            "/tmp/project".to_string(),
            mode,
            None,
            plain_manager(),
            offline_registry().await,
            Arc::new(|| Ok(())),
            Arc::new(ContextActions::default()),
        );
        assert_eq!(
            call_context(&context, "mode", None, false, &mut |_signal| 1).expect("mode"),
            json!(name),
        );
    }
}

#[tokio::test]
async fn unknown_domain_names_and_bad_params_answer_in_scope() {
    let context = test_context(ContextActions::default()).await;
    let error = call_context(&context, "session.nope", None, false, &mut |_signal| 1)
        .expect_err("unknown domain method");
    assert_eq!(error.code(), "unknown_method");
    let error = call_context(
        &context,
        "compact",
        Some(&json!("not an object")),
        false,
        &mut |_signal| 1,
    )
    .expect_err("bad params");
    assert_eq!(error.code(), "invalid_params");
}

#[tokio::test]
async fn debug_views_carry_the_bound_shape() {
    let context = test_context(ContextActions::default()).await;
    assert!(format!("{context:?}").contains("HandlerContext"));
    let options = HostConnectOptions {
        path: "/ext/demo".to_string(),
        resolved_path: "/ext/demo".to_string(),
        source_info: source_info(),
        context: shared_context(context),
        ui: None,
        error_listener: None,
    };
    let debug = format!("{options:?}");
    assert!(debug.contains("/ext/demo"));
    assert!(debug.contains("has_ui: false"));
}

#[tokio::test]
async fn context_calls_answer_unsupported_before_a_context_is_bound() {
    let mut options = support::connect_options(ContextActions::default(), None).await;
    options.context = crate::extensions::host::SharedContext::default();
    let (host_io, child_io) = tokio::io::duplex(1 << 20);
    let task = tokio::spawn(async move { HostSession::connect(host_io, options).await });
    let mut child = Link::new(child_io).expect("child link");
    child.write_frame(&hello()).await.expect("hello");
    assert!(matches!(
        child.read_frame().await.expect("welcome"),
        Some(Frame::Welcome { .. })
    ));
    child
        .write_frame(&Frame::RegisterEvent {
            channel: "agent_start".to_string(),
        })
        .await
        .expect("register");
    child.write_frame(&Frame::Ready).await.expect("ready");
    let session = task.await.expect("join").expect("connect");

    let handler = handler(&session, "agent_start");
    let context = test_context(ContextActions::default()).await;
    let dispatch = tokio::spawn(async move { handler(ExtensionEvent::AgentStart, context).await });
    let Some(Frame::CallEvent { call, .. }) = child.read_frame().await.expect("call") else {
        panic!("the host sends a call event");
    };
    child
        .write_frame(&Frame::ContextCall {
            call,
            id: 1,
            method: "cwd".to_string(),
            params: None,
        })
        .await
        .expect("context call");
    let Some(Frame::ContextError {
        call: owner,
        id,
        code,
        ..
    }) = child.read_frame().await.expect("answer")
    else {
        panic!("the answer is an in-scope error");
    };
    assert_eq!((owner, id), (call, 1));
    assert_eq!(code, "unsupported");
    child
        .write_frame(&Frame::CallResult {
            call,
            result: Value::Null,
        })
        .await
        .expect("result");
    assert!(matches!(dispatch.await.expect("join"), HandlerResult::None));
}
