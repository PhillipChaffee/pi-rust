#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use serde_json::{Value, json};

use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::{
    CustomMessagePayload, ExtensionMode, InputSource, SessionShutdownReason, StreamingBehavior,
    text_block,
};
use crate::extensions::results::{
    HandlerResult, InputResult, MessageEndResult, ProjectTrustDecision, ProjectTrustResult,
    SessionBeforeResult, ToolCallResult, ToolResultResult,
};
use crate::extensions::runner::{
    CommandContextActions, DeliverAs, ExtensionActions, ExtensionRunner, ForkOptions, ForkPosition,
    NavigateTreeOptions, NewSessionOptions, ProjectTrustOutcome, ProviderActions,
    SendMessageOptions, SendUserMessageOptions, SessionReplaceOutcome, SwitchSessionOptions,
    emit_project_trust_event, emit_session_shutdown_event,
};
use crate::extensions::runtime::{
    AssertActiveFn, ContextActions, Extension, ExtensionError, ExtensionFlag, ExtensionHandler,
    FlagType, FlagValue, HandlerContext,
};
use crate::model_registry::ModelRegistry;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::provider_composer::ProviderConfigInput;
use crate::session_manager::SessionManager;
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};
use crate::tools::bash::create_bash_tool_definition;
use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::credential_store::InMemoryCredentialStore;
use pi_ai::types::{Api, Model, ProviderId, Usage, UserContent};

type EventLog = Arc<Mutex<Vec<&'static str>>>;

fn source_info() -> SourceInfo {
    crate::source_info::create_synthetic_source_info(
        "/ext/test.ts",
        &SyntheticSourceOptions {
            source: "extension:test".to_string(),
            scope: Some(SourceScope::Temporary),
            origin: Some(SourceOrigin::TopLevel),
            base_dir: None,
        },
    )
}

fn plain_manager() -> SessionManager {
    SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
        .expect("the in-memory manager constructs")
}

async fn offline_registry() -> ModelRegistry {
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::default())),
        models_path: Some(None),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs offline");
    ModelRegistry::new(runtime)
}

async fn runner_with(
    extensions: &[Arc<Extension>],
    actions: Arc<ContextActions>,
) -> ExtensionRunner {
    ExtensionRunner::new(
        extensions.to_vec(),
        actions,
        Arc::new(|| Ok(())),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        crate::extensions::api::ExtensionRuntime::new(),
    )
}

fn plain_actions() -> Arc<ContextActions> {
    Arc::new(ContextActions::default())
}

fn logging_handler(log: &EventLog, tag: &'static str, result: HandlerResult) -> ExtensionHandler {
    let log = Arc::clone(log);
    Arc::new(move |_event: ExtensionEvent, _ctx: HandlerContext| {
        let log = Arc::clone(&log);
        let result = result.clone();
        Box::pin(async move {
            log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(tag);
            result
        })
    })
}

fn extension_with(entries: Vec<(&'static str, ExtensionHandler)>) -> Arc<Extension> {
    let extension = Extension::new(
        "/ext/test.ts".to_string(),
        "/ext/test.ts".to_string(),
        false,
        source_info(),
    );
    for (channel, handler) in entries {
        extension.on(channel, handler);
    }
    Arc::new(extension)
}
fn assistant_message() -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::Assistant(
        pi_ai::types::AssistantMessage {
            content: Vec::new(),
            api: Api::from("openai"),
            provider: ProviderId::from("openai"),
            model: "gpt-test".to_string(),
            response_model: None,
            response_id: None,
            provider_thinking_level: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: pi_ai::types::StopReason::Stop,
            deferred: None,
            error_message: None,
            raw_stop_reason: None,
            end_turn: None,
            timestamp: 0,
        },
    ))
}

fn user_message() -> AgentMessage {
    AgentMessage::Standard(pi_ai::types::Message::User(pi_ai::types::UserMessage {
        content: UserContent::Text(String::new()),
        timestamp: 0,
    }))
}

fn tool_result_text(text: &str) -> pi_ai::types::ToolResultBlock {
    pi_ai::types::ToolResultBlock::Text(pi_ai::types::TextContent {
        text: text.to_string(),
        text_signature: None,
    })
}

#[tokio::test]
async fn emit_walks_extensions_then_handlers_and_yields_none_outside_session_before() {
    let log: EventLog = Arc::new(Mutex::new(Vec::new()));
    let first = extension_with(vec![
        (
            "agent_start",
            logging_handler(&log, "e1-h1", HandlerResult::None),
        ),
        (
            "agent_start",
            logging_handler(&log, "e1-h2", HandlerResult::None),
        ),
    ]);
    let second = extension_with(vec![(
        "agent_start",
        logging_handler(&log, "e2-h1", HandlerResult::None),
    )]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner.emit(ExtensionEvent::AgentStart).await;
    assert!(matches!(outcome, SessionBeforeResult::None));
    assert_eq!(
        *log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec!["e1-h1", "e1-h2", "e2-h1"]
    );
}

#[tokio::test]
async fn emit_merges_session_before_last_defined_result_wins() {
    let first = extension_with(vec![(
        "session_before_switch",
        always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
            cancel: false,
        })),
    )]);
    let second = extension_with(vec![(
        "session_before_switch",
        always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
            cancel: false,
        })),
    )]);
    // The following replacement also wins over the earlier result.
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner
        .emit(ExtensionEvent::SessionBeforeSwitch {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        })
        .await;
    assert!(
        matches!(outcome, SessionBeforeResult::Switch { cancel: false }),
        "{outcome:?}"
    );
}

fn always_handler(result: HandlerResult) -> ExtensionHandler {
    Arc::new(move |_event: ExtensionEvent, _ctx: HandlerContext| {
        let result = result.clone();
        Box::pin(async move { result })
    })
}

#[tokio::test]
async fn emit_cancel_short_circuits_the_remaining_handlers() {
    let log: EventLog = Arc::new(Mutex::new(Vec::new()));
    let first = extension_with(vec![(
        "session_before_switch",
        always_handler(HandlerResult::SessionBefore(SessionBeforeResult::Switch {
            cancel: true,
        })),
    )]);
    let second = extension_with(vec![(
        "session_before_switch",
        logging_handler(&log, "skipped", HandlerResult::None),
    )]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner
        .emit(ExtensionEvent::SessionBeforeSwitch {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        })
        .await;
    assert!(
        matches!(outcome, SessionBeforeResult::Switch { cancel: true }),
        "{outcome:?}"
    );
    assert_eq!(
        *log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Vec::<&str>::new()
    );
}

#[tokio::test]
async fn emit_tool_result_merges_fields_across_handlers() {
    let content_patch = HandlerResult::ToolResult(ToolResultResult {
        content: Some(vec![tool_result_text("patched")]),
        ..ToolResultResult::default()
    });
    let error_patch = HandlerResult::ToolResult(ToolResultResult {
        is_error: Some(true),
        ..ToolResultResult::default()
    });
    let first = extension_with(vec![("tool_result", always_handler(content_patch))]);
    let second = extension_with(vec![("tool_result", always_handler(error_patch))]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner
        .emit_tool_result(ExtensionEvent::ToolResult {
            tool_name: "read".to_string(),
            tool_call_id: "call-1".to_string(),
            input: json!({ "path": "/x" }),
            content: vec![tool_result_text("original")],
            details: json!({ "ok": true }),
            is_error: false,
            usage: None,
        })
        .await;
    let merged = outcome.expect("a handler modified the result");
    assert_eq!(
        merged.content.as_deref(),
        Some(&[tool_result_text("patched")][..])
    );
    assert_eq!(merged.is_error, Some(true));
    assert_eq!(merged.details, Some(json!({ "ok": true })));
}

#[tokio::test]
async fn emit_tool_result_yields_none_without_modification() {
    let first = extension_with(vec![("tool_result", always_handler(HandlerResult::None))]);
    let runner = runner_with(&[first], plain_actions()).await;

    let outcome = runner
        .emit_tool_result(ExtensionEvent::ToolResult {
            tool_name: "read".to_string(),
            tool_call_id: "call-1".to_string(),
            input: Value::Null,
            content: Vec::new(),
            details: Value::Null,
            is_error: false,
            usage: None,
        })
        .await;
    assert!(outcome.is_none());
}

#[tokio::test]
async fn emit_tool_call_block_short_circuits_and_last_defined_wins() {
    let log: EventLog = Arc::new(Mutex::new(Vec::new()));
    let unblock = always_handler(HandlerResult::ToolCall(ToolCallResult {
        block: false,
        ..ToolCallResult::default()
    }));
    let block = always_handler(HandlerResult::ToolCall(ToolCallResult {
        block: true,
        reason: Some("blocked".to_string()),
        ..ToolCallResult::default()
    }));
    let first = extension_with(vec![("tool_call", unblock)]);
    let second = extension_with(vec![
        ("tool_call", block.clone()),
        (
            "tool_call",
            logging_handler(&log, "after-block", HandlerResult::None),
        ),
    ]);
    let runner = runner_with(&[Arc::clone(&first), second], plain_actions()).await;

    let outcome = runner
        .emit_tool_call(ExtensionEvent::ToolCall {
            tool_name: "bash".to_string(),
            tool_call_id: "call-1".to_string(),
            input: json!({ "command": "rm -rf /" }),
        })
        .await;
    let blocked = outcome.expect("the block short-circuits");
    assert!(blocked.block);
    assert_eq!(blocked.reason.as_deref(), Some("blocked"));
    assert_eq!(
        *log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Vec::<&str>::new()
    );

    // Without a block the last defined result wins.
    let last = always_handler(HandlerResult::ToolCall(ToolCallResult {
        block: false,
        reason: Some("last".to_string()),
        ..ToolCallResult::default()
    }));
    let replace = extension_with(vec![("tool_call", last)]);
    let runner = runner_with(&[Arc::clone(&first), replace], plain_actions()).await;
    let outcome = runner
        .emit_tool_call(ExtensionEvent::ToolCall {
            tool_name: "bash".to_string(),
            tool_call_id: "call-2".to_string(),
            input: Value::Null,
        })
        .await;
    assert_eq!(
        outcome.and_then(|resolved| resolved.reason),
        Some("last".to_string())
    );
}

#[tokio::test]
async fn emit_user_bash_first_decision_wins() {
    let first = extension_with(vec![(
        "user_bash",
        always_handler(HandlerResult::UserBash(
            crate::extensions::results::UserBashResult { handled: true },
        )),
    )]);
    let second = extension_with(vec![(
        "user_bash",
        always_handler(HandlerResult::UserBash(
            crate::extensions::results::UserBashResult { handled: false },
        )),
    )]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner.emit_user_bash("cargo test".to_string()).await;
    assert_eq!(outcome.map(|resolved| resolved.handled), Some(true));
}
#[tokio::test]
async fn emit_context_chains_message_replacements() {
    let replace_with_user = |tag: &'static str| {
        let replacement = user_message_with(tag);
        always_handler(HandlerResult::Context(
            crate::extensions::results::ContextResult {
                messages: Some(vec![replacement]),
            },
        ))
    };
    let first = extension_with(vec![("context", replace_with_user("first"))]);
    let second = extension_with(vec![("context", replace_with_user("second"))]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let outcome = runner.emit_context(vec![assistant_message()]).await;
    assert_eq!(outcome, vec![user_message_with("second")]);
}

fn user_message_with(tag: &'static str) -> AgentMessage {
    AgentMessage::Custom(pi_agent_core::types::CustomAgentMessage {
        role: tag.to_string(),
        timestamp: 0,
        data: serde_json::Map::new(),
    })
}

#[tokio::test]
async fn emit_before_provider_request_chains_replacements() {
    let first = extension_with(vec![(
        "before_provider_request",
        always_handler(HandlerResult::BeforeProviderRequest(json!({
            "model": "step-1"
        }))),
    )]);
    let second = extension_with(vec![(
        "before_provider_request",
        always_handler(HandlerResult::BeforeProviderRequest(json!({
            "model": "step-2"
        }))),
    )]);
    let runner = runner_with(&[Arc::clone(&first), second], plain_actions()).await;

    let outcome = runner
        .emit_before_provider_request(json!({ "model": "original" }))
        .await;
    assert_eq!(outcome, json!({ "model": "step-2" }));

    // A handler returning none keeps the running payload.
    let passthrough = extension_with(vec![(
        "before_provider_request",
        always_handler(HandlerResult::None),
    )]);
    let runner = runner_with(&[Arc::clone(&first), passthrough], plain_actions()).await;
    let outcome = runner
        .emit_before_provider_request(json!({ "model": "original" }))
        .await;
    assert_eq!(outcome, json!({ "model": "step-1" }));
}
#[tokio::test]
async fn emit_before_provider_headers_restates_mutations() {
    let headers_renamed = always_handler(HandlerResult::BeforeProviderHeaders(
        pi_ai::types::ProviderHeaders::new(),
    ));
    let first = extension_with(vec![("before_provider_headers", headers_renamed)]);
    let passthrough = extension_with(vec![(
        "before_provider_headers",
        always_handler(HandlerResult::None),
    )]);
    let runner = runner_with(&[first, passthrough], plain_actions()).await;

    let outcome = runner
        .emit_before_provider_headers(pi_ai::types::ProviderHeaders::new())
        .await;
    assert!(format!("{outcome:?}").is_empty() || !format!("{outcome:?}").contains("Authorization"));
}

#[tokio::test]
async fn emit_before_agent_start_chains_prompt_and_queues_messages() {
    let first = extension_with(vec![(
        "before_agent_start",
        always_handler(HandlerResult::BeforeAgentStart(
            crate::extensions::results::BeforeAgentStartResult {
                message: Some(custom_payload("one")),
                system_prompt: Some("prompt-1".to_string()),
            },
        )),
    )]);
    let second = extension_with(vec![(
        "before_agent_start",
        always_handler(HandlerResult::BeforeAgentStart(
            crate::extensions::results::BeforeAgentStartResult {
                message: Some(custom_payload("two")),
                system_prompt: Some("prompt-2".to_string()),
            },
        )),
    )]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let combined = runner
        .emit_before_agent_start(
            "ship it".to_string(),
            None,
            "base".to_string(),
            crate::system_prompt::BuildSystemPromptOptions::default(),
        )
        .await
        .expect("the handlers contributed");
    assert_eq!(combined.messages.len(), 2);
    assert_eq!(combined.system_prompt.as_deref(), Some("prompt-2"));
}

#[tokio::test]
async fn emit_before_agent_start_context_reads_the_running_prompt() {
    let reads_prompt: ExtensionHandler = Arc::new(|_event: ExtensionEvent, ctx: HandlerContext| {
        Box::pin(async move {
            let prompt = ctx.get_system_prompt().unwrap_or_default();
            HandlerResult::BeforeAgentStart(crate::extensions::results::BeforeAgentStartResult {
                message: None,
                system_prompt: if prompt == "running" {
                    Some("after".to_string())
                } else {
                    None
                },
            })
        })
    });
    let first = extension_with(vec![(
        "before_agent_start",
        always_handler(HandlerResult::BeforeAgentStart(
            crate::extensions::results::BeforeAgentStartResult {
                message: None,
                system_prompt: Some("running".to_string()),
            },
        )),
    )]);
    let second = extension_with(vec![("before_agent_start", reads_prompt)]);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let combined = runner
        .emit_before_agent_start(
            "ship it".to_string(),
            None,
            "base".to_string(),
            crate::system_prompt::BuildSystemPromptOptions::default(),
        )
        .await
        .expect("the prompt replacement is a contribution");
    assert_eq!(combined.system_prompt.as_deref(), Some("after"));
    assert!(combined.messages.is_empty());
}

fn custom_payload(tag: &'static str) -> CustomMessagePayload {
    CustomMessagePayload {
        custom_type: tag.to_string(),
        content: vec![text_block(tag)],
        display: true,
        details: None,
    }
}
#[tokio::test]
async fn emit_resources_discover_carries_provenance_and_skips_empty() {
    let first = extension_with(vec![(
        "resources_discover",
        always_handler(HandlerResult::ResourcesDiscover(
            crate::extensions::results::ResourcesDiscoverResult {
                skill_paths: vec!["/skills/a".to_string()],
                ..crate::extensions::results::ResourcesDiscoverResult::default()
            },
        )),
    )]);
    let second = extension_with(vec![(
        "resources_discover",
        always_handler(HandlerResult::ResourcesDiscover(
            crate::extensions::results::ResourcesDiscoverResult {
                prompt_paths: vec!["/prompts/b".to_string()],
                ..crate::extensions::results::ResourcesDiscoverResult::default()
            },
        )),
    )]);
    second.set_hidden(true);
    let runner = runner_with(&[first, second], plain_actions()).await;

    let paths = runner
        .emit_resources_discover(
            "/tmp/project".to_string(),
            crate::extensions::reasons::SessionStartReason::Startup,
        )
        .await;
    assert_eq!(paths.skill_paths.len(), 1);
    assert_eq!(paths.skill_paths[0].path, "/skills/a");
    assert_eq!(paths.skill_paths[0].extension_path, "/ext/test.ts");
    assert_eq!(paths.prompt_paths.len(), 1);
    assert_eq!(paths.prompt_paths[0].path, "/prompts/b");
    assert!(paths.theme_paths.is_empty());
    let rendered = format!("{runner:?}");
    assert!(rendered.contains("ExtensionRunner"), "{rendered}");
}

#[tokio::test]
async fn emit_input_chains_transforms_and_short_circuits_on_handled() {
    let transform = HandlerResult::Input(InputResult::Transform {
        text: "expanded".to_string(),
        images: None,
    });
    let first = extension_with(vec![("input", always_handler(transform))]);
    let handled = extension_with(vec![(
        "input",
        always_handler(HandlerResult::Input(InputResult::Handled)),
    )]);
    let runner = runner_with(&[Arc::clone(&first), handled], plain_actions()).await;
    assert!(matches!(
        runner
            .emit_input("hi".to_string(), None, InputSource::Interactive, None)
            .await,
        InputResult::Handled
    ));

    // A pure-changed chain restates as a final transform.
    let second_transform = HandlerResult::Input(InputResult::Transform {
        text: "expanded-twice".to_string(),
        images: Some(Vec::new()),
    });
    let second = extension_with(vec![("input", always_handler(second_transform))]);
    let runner = runner_with(&[first, second], plain_actions()).await;
    assert!(matches!(
        runner
            .emit_input("hi".to_string(), None, InputSource::Rpc, Some(StreamingBehavior::FollowUp))
            .await,
        InputResult::Transform {
            text,
            images: Some(_)
        } if text == "expanded-twice"
    ));

    // No handler action restates as continue.
    let unbound_runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    assert!(matches!(
        unbound_runner
            .emit_input("hi".to_string(), None, InputSource::Interactive, None)
            .await,
        InputResult::Continue
    ));
}
#[tokio::test]
async fn emit_message_end_chains_and_rejects_role_changes() {
    let errors: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
    let replacement = HandlerResult::MessageEnd(MessageEndResult {
        message: Some(assistant_message()),
    });
    let first = extension_with(vec![("message_end", always_handler(replacement))]);
    let role_swap = HandlerResult::MessageEnd(MessageEndResult {
        message: Some(user_message()),
    });
    let second = extension_with(vec![("message_end", always_handler(role_swap))]);
    let runner = runner_with(&[first, second], plain_actions()).await;
    let _unsubscribe = runner.on_error(Arc::new({
        let errors = Arc::clone(&errors);
        move |error: &ExtensionError| {
            errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(error.clone());
        }
    }));

    let outcome = runner.emit_message_end(assistant_message()).await;
    assert!(outcome.is_some(), "the same-role replacement chains");
    let recorded = Arc::clone(&errors)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(
        recorded[0].error,
        "message_end handlers must return a message with the same role"
    );
    assert_eq!(recorded[0].event, "message_end");
    assert_eq!(recorded[0].extension_path, "/ext/test.ts");
}

#[tokio::test]
async fn on_error_unsubscribe_stops_listening() {
    let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    // The unbound active-tool reader answers empty, the default the
    // wrapper reads before a bind lands.
    assert_eq!(
        runner.active_tools().expect("the unbound reader"),
        Vec::<String>::new()
    );
    let hits: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let listener_hits = Arc::clone(&hits);
    let unsubscribe = runner.on_error(Arc::new({
        let listener_hits = Arc::clone(&listener_hits);
        move |_: &ExtensionError| {
            listener_hits
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push("one");
        }
    }));
    runner.emit_error(&ExtensionError {
        extension_path: "/ext/test.ts".to_string(),
        event: "session_start".to_string(),
        error: "boom".to_string(),
        stack: None,
    });
    assert_eq!(
        hits.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
    unsubscribe();
    runner.emit_error(&ExtensionError {
        extension_path: "/ext/test.ts".to_string(),
        event: "session_start".to_string(),
        error: "boom".to_string(),
        stack: None,
    });
    assert_eq!(
        hits.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
}

#[tokio::test]
async fn emit_session_shutdown_event_gates_on_registered_handlers() {
    let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    let emitted = emit_session_shutdown_event(
        &runner,
        ExtensionEvent::SessionShutdown {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        },
    )
    .await;
    assert!(!emitted, "no handlers registered");

    let with_handlers = extension_with(vec![(
        "session_shutdown",
        always_handler(HandlerResult::None),
    )]);
    let runner = runner_with(&[with_handlers], plain_actions()).await;
    let emitted = emit_session_shutdown_event(
        &runner,
        ExtensionEvent::SessionShutdown {
            reason: SessionShutdownReason::Quit,
            target_session_file: None,
        },
    )
    .await;
    assert!(emitted, "the handler count drives the gate");
}

#[tokio::test]
async fn emit_project_trust_first_decision_wins_undecided_falls_through() {
    let undecided = extension_with(vec![(
        "project_trust",
        always_handler(HandlerResult::ProjectTrust(ProjectTrustResult::default())),
    )]);
    let deny = extension_with(vec![(
        "project_trust",
        always_handler(HandlerResult::ProjectTrust(ProjectTrustResult {
            trusted: Some(ProjectTrustDecision::No),
        })),
    )]);
    let context = HandlerContext::new(
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        always_active(),
        plain_actions(),
    );
    let outcome = emit_project_trust_decision(std::slice::from_ref(&deny), context.clone()).await;
    assert_eq!(
        outcome.result.and_then(|resolved| resolved.trusted),
        Some(ProjectTrustDecision::No)
    );
    assert!(outcome.errors.is_empty());

    let undecided_runner =
        emit_project_trust_decision(std::slice::from_ref(&undecided), context).await;
    assert!(undecided_runner.result.is_none());
    assert_eq!(undecided_runner.errors.len(), 0);
}

async fn emit_project_trust_decision(
    extensions: &[Arc<Extension>],
    context: HandlerContext,
) -> ProjectTrustOutcome {
    emit_project_trust_event(
        extensions,
        ExtensionEvent::ProjectTrust {
            cwd: "/tmp/project".to_string(),
        },
        context,
    )
    .await
}

fn always_active() -> AssertActiveFn {
    Arc::new(|| Ok(()))
}

/// The bind-core bundle with inert handlers, the base tests re-arm
/// individual slots over.
fn trivial_actions() -> ExtensionActions {
    ExtensionActions {
        send_message: Arc::new(
            |_message: CustomMessagePayload, _options: Option<SendMessageOptions>| {},
        ),
        send_user_message: Arc::new(
            |_content: UserContent, _options: Option<SendUserMessageOptions>| {},
        ),
        append_entry: Arc::new(|_custom_type: String, _data: Option<Value>| {}),
        set_session_name: Arc::new(|_name: String| {}),
        get_session_name: Arc::new(|| None),
        set_label: Arc::new(|_entry_id: String, _label: Option<String>| {}),
        get_active_tools: Arc::new(Vec::new),
        get_all_tools: Arc::new(Vec::new),
        set_active_tools: Arc::new(|_tool_names: Vec<String>| {}),
        refresh_tools: Arc::new(|| {}),
        get_commands: Arc::new(Vec::new),
        set_model: Arc::new(|_model: Model| Box::pin(async { false })),
        get_thinking_level: Arc::new(|| ThinkingLevel::Off),
        set_thinking_level: Arc::new(|_level: ThinkingLevel| {}),
    }
}

type ProviderRecords = (
    ProviderActions,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
);

fn inert_provider_actions() -> ProviderRecords {
    let registered: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let unregistered: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    (
        ProviderActions {
            register_provider: {
                let registered = Arc::clone(&registered);
                Arc::new(move |name: String, _config: ProviderConfigInput| {
                    registered
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(name);
                    Ok(())
                })
            },
            unregister_provider: {
                let unregistered = Arc::clone(&unregistered);
                Arc::new(move |name: String| {
                    unregistered
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push(name);
                    Ok(())
                })
            },
        },
        registered,
        unregistered,
    )
}

/// An empty registered-flag table, the tests' first-wins aggregate over.
fn plain_flag(name: &str) -> ExtensionFlag {
    ExtensionFlag {
        name: name.to_string(),
        description: None,
        flag_type: FlagType::String,
        default: None,
        extension_path: "/ext/test.ts".to_string(),
    }
}

fn runtime_with_flag(name: &str, value: FlagValue) -> crate::extensions::api::ExtensionRuntime {
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    runtime.set_flag_value(name, value);
    runtime
}

/// A runner over the given extensions and runtime, the
/// non-default-runtime boundary tests read through.
async fn runner_with_runtime(
    extensions: &[Arc<Extension>],
    runtime: crate::extensions::api::ExtensionRuntime,
) -> ExtensionRunner {
    ExtensionRunner::new(
        extensions.to_vec(),
        plain_actions(),
        always_active(),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        runtime,
    )
}

#[tokio::test]
async fn bound_actions_read_after_bind_and_fail_before() {
    let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    assert!(
        runner
            .bound_actions()
            .expect_err("the unbound bundle fails")
            .contains("not bound")
    );

    let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    runner.bind_core(trivial_actions(), None);
    let bundle = runner.bound_actions().expect("the bound bundle reads back");
    (bundle.send_message)(
        CustomMessagePayload {
            custom_type: "note".to_string(),
            content: vec![text_block("blog")],
            display: true,
            details: None,
        },
        Some(SendMessageOptions {
            trigger_turn: Some(true),
            deliver_as: Some(DeliverAs::FollowUp),
        }),
    );
    (bundle.send_user_message)(
        UserContent::Text("ping".to_string()),
        Some(SendUserMessageOptions {
            deliver_as: Some(DeliverAs::Steer),
            expand_prompt_templates: Some(true),
        }),
    );
    (bundle.append_entry)("entries".to_string(), Some(json!({ "kind": 1 })));
    (bundle.set_session_name)("session-name".to_string());
    assert!((bundle.get_session_name)().is_none());
    (bundle.set_label)("entry-1".to_string(), Some("label".to_string()));
    assert!((bundle.get_active_tools)().is_empty());
    assert!((bundle.get_all_tools)().is_empty());
    (bundle.set_active_tools)(vec!["read".to_string()]);
    (bundle.refresh_tools)();
    assert!((bundle.get_commands)().is_empty());
    let refused = (bundle.set_model)(trivial_model()).await;
    assert!(!refused);
    assert_eq!((bundle.get_thinking_level)(), ThinkingLevel::Off);
    (bundle.set_thinking_level)(ThinkingLevel::High);
    assert!(
        runner.active_tools().expect("the bound reader").is_empty(),
        "the bind arms the wrapper's reader too"
    );
}

/// The model literal the set-model bind exercises; no registry lookup.
fn trivial_model() -> Model {
    Model {
        id: "gpt-test".to_string(),
        name: "GPT Test".to_string(),
        api: Api::from("openai"),
        provider: ProviderId::from("openai"),
        base_url: "https://example.invalid".to_string(),
        reasoning: false,
        thinking_level_map: None,
        input: Vec::new(),
        cost: pi_ai::types::ModelCost::default(),
        context_window: 0,
        max_tokens: 0,
        sampling_params: None,
        headers: None,
        compat: None,
    }
}

#[tokio::test]
async fn bind_core_flushes_the_provider_queue() {
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
        name: "queued-provider".to_string(),
        config: ProviderConfigInput::default(),
        extension_path: "/ext/test.ts".to_string(),
    });

    // The actions flush through the caller's provider actions.
    let (provider_actions, registered, _unregistered) = inert_provider_actions();
    let rendered = format!("{provider_actions:?}");
    assert!(rendered.contains("ProviderActions"));
    let mut runner = ExtensionRunner::new(
        Vec::new(),
        plain_actions(),
        always_active(),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        runtime.clone(),
    );
    runner.bind_core(trivial_actions(), Some(provider_actions));
    let armed = runtime
        .bound_provider_actions()
        .expect("the flush arms the immediate path");
    (armed.unregister_provider)("queued-provider".to_string())
        .expect("the armed unregister reads through");
    assert_eq!(
        *registered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec!["queued-provider".to_string()]
    );
    assert!(
        runtime.take_pending_provider_registrations().is_empty(),
        "the queue drains flush-first"
    );

    // Without provider actions the flush lands on the model registry.
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
        name: "registered-provider".to_string(),
        config: ProviderConfigInput::default(),
        extension_path: "/ext/test.ts".to_string(),
    });
    let mut runner = ExtensionRunner::new(
        Vec::new(),
        plain_actions(),
        always_active(),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        runtime.clone(),
    );
    runner.bind_core(trivial_actions(), None);
    assert!(
        runner
            .model_registry
            .get_registered_provider_config("registered-provider")
            .is_some(),
        "the queue flushed into the registry"
    );
}

#[tokio::test]
async fn flush_errors_report_to_the_listeners() {
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    let runtime_for_armed = runtime.clone();
    runtime.register_provider(crate::extensions::api::PendingProviderRegistration {
        name: "failing-provider".to_string(),
        config: ProviderConfigInput::default(),
        extension_path: "/ext/failing.ts".to_string(),
    });
    let mut runner = ExtensionRunner::new(
        Vec::new(),
        plain_actions(),
        always_active(),
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        plain_manager(),
        offline_registry().await,
        runtime,
    );
    let failed: Arc<Mutex<Vec<ExtensionError>>> = Arc::new(Mutex::new(Vec::new()));
    let _unsubscribe = runner.on_error(Arc::new({
        let failed = Arc::clone(&failed);
        move |error: &ExtensionError| {
            failed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(error.clone());
        }
    }));
    runner.bind_core(
        trivial_actions(),
        Some(ProviderActions {
            register_provider: Arc::new(|name: String, _config: ProviderConfigInput| {
                Err(format!("cannot register {name}"))
            }),
            unregister_provider: Arc::new(|_name: String| Ok(())),
        }),
    );
    // The armed actions read through the runtime too.
    let armed = runtime_for_armed.bound_provider_actions().expect("armed");
    (armed.unregister_provider)("failing-provider".to_string()).expect("the armed unregister");
    let failed = failed
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].extension_path, "/ext/failing.ts");
    assert_eq!(failed[0].event, "register_provider");
    assert_eq!(failed[0].error, "cannot register failing-provider");
}

#[tokio::test]
async fn flags_aggregate_first_registration_wins_and_values_roundtrip() {
    let first = extension_with(Vec::new());
    first.register_flag(plain_flag("alpha"));
    let second = extension_with(Vec::new());
    second.register_flag(plain_flag("alpha"));
    second.register_flag(plain_flag("beta"));
    let runtime = runtime_with_flag("alpha", FlagValue::Bool(true));
    let runner = runner_with_runtime(&[first, second], runtime).await;
    let names = runner
        .get_flags()
        .into_iter()
        .map(|flag| flag.name)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["alpha", "beta"],
        "first per name wins in load order"
    );

    runner.set_flag_value("alpha", FlagValue::Text("late".to_string()));
    assert_eq!(
        runner.get_flag_values().get("alpha"),
        Some(&FlagValue::Text("late".to_string()))
    );
}

fn command(name: &str) -> crate::extensions::runtime::ResolvedCommand {
    crate::extensions::runtime::ResolvedCommand {
        name: name.to_string(),
        invocation_name: name.to_string(),
        description: None,
        source_info: source_info(),
        handler: Arc::new(|_args: String, _ctx| Box::pin(async {})),
    }
}

#[tokio::test]
async fn registered_commands_rename_collisions_and_land_free_names() {
    let first = extension_with(Vec::new());
    first.register_command(command("ship"));
    first.register_command(command("alpha"));
    let second = extension_with(Vec::new());
    second.register_command(command("ship"));
    second.register_command(command("ship:1"));
    let runner = runner_with(&[first, second], plain_actions()).await;

    let names = runner
        .get_registered_commands()
        .into_iter()
        .map(|resolved| resolved.invocation_name)
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "ship:1".to_string(),
            "alpha".to_string(),
            "ship:2".to_string(),
            "ship:1:2".to_string(),
        ],
        "collisions take name:occurrence and evicted invocations walk the suffixes"
    );

    let found = runner
        .get_command("ship:2")
        .expect("the second ship resolves by its renamed invocation");
    assert_eq!(found.name, "ship");
    (found.handler)("args".to_string(), runner.context()).await;
    assert!(runner.get_command("ship").is_none());

    // A unique name keeps its bare name and the reads reset diagnostics.
    let single = extension_with(Vec::new());
    single.register_command(command("only"));
    let runner = runner_with(&[single], plain_actions()).await;
    assert_eq!(
        runner.get_command("only").expect("bare name").name,
        "only".to_string()
    );
    assert!(runner.get_command_diagnostics().is_empty());
    let _ = runner.get_registered_commands();
    assert!(
        runner.get_command_diagnostics().is_empty(),
        "the read resets"
    );
}

#[tokio::test]
async fn all_registered_tools_keep_the_first_registration() {
    let renamed = |name: &str| {
        let mut definition = create_bash_tool_definition(".", None);
        definition.name = name.to_string();
        definition.label = name.to_string();
        crate::extensions::runtime::RegisteredTool {
            definition,
            source_info: source_info(),
        }
    };
    let first = extension_with(Vec::new());
    first.register_tool(renamed("shared"));
    first.register_tool(renamed("first-only"));
    let second = extension_with(Vec::new());
    second.register_tool(renamed("shared"));
    let runner = runner_with(&[first, second], plain_actions()).await;

    let names = runner
        .get_all_registered_tools()
        .into_iter()
        .map(|tool| tool.definition.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["shared".to_string(), "first-only".to_string()]);

    let resolved = runner
        .get_tool_definition("shared")
        .expect("the tool read resolves")
        .expect("the first registration wins");
    assert_eq!(resolved.name, "shared".to_string());
    assert!(
        runner
            .get_tool_definition("missing")
            .expect("the tool read resolves")
            .is_none()
    );
}

#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the bound/default/reset ladder walks every command-context handler"
)]
async fn command_context_defaults_and_bound_handlers_route() {
    let runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    assert_eq!(
        runner.get_extension_paths(),
        vec!["/ext/test.ts".to_string()]
    );

    // The unbound defaults answer the no-op contracts for all six.
    let context = runner.command_context();
    let rendered = format!("{context:?}");
    assert!(rendered.contains("ExtensionCommandContext"));
    assert!(
        !context
            .new_session(NewSessionOptions::default())
            .await
            .expect("the default replacement")
            .cancelled
    );
    context
        .wait_for_idle()
        .await
        .expect("the default idle wait");
    assert!(
        !context
            .fork("entry-0".to_string(), ForkOptions::default())
            .await
            .expect("the default fork")
            .cancelled
    );
    assert!(
        !context
            .navigate_tree("target-0".to_string(), NavigateTreeOptions::default())
            .await
            .expect("the default navigation")
            .cancelled
    );
    assert!(
        !context
            .switch_session("/session.json".to_string(), SwitchSessionOptions::default(),)
            .await
            .expect("the default switch")
            .cancelled
    );
    context.reload().await.expect("the default reload");
    assert_eq!(
        context.get_system_prompt_options().expect("options").cwd,
        "/tmp/project".to_string()
    );
    let rendered = format!("{runner:?}");
    assert!(rendered.contains("ExtensionRunner"));

    // The bound handlers route with their arguments.
    let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    let command_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let hold_idle = Arc::clone(&command_log);
    let hold_fork = Arc::clone(&command_log);
    let hold_nav = Arc::clone(&command_log);
    let hold_switch = Arc::clone(&command_log);
    let hold_reload = Arc::clone(&command_log);
    let waitlog = Arc::clone(&command_log);
    let fork_entry: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let entry_read = Arc::clone(&fork_entry);
    let position_read: Arc<Mutex<Option<ForkPosition>>> = Arc::new(Mutex::new(None));
    let position_read2 = Arc::clone(&position_read);
    runner.bind_command_context(Some(CommandContextActions {
        wait_for_idle: Arc::new(move || {
            let log = Arc::clone(&waitlog);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("idle".to_string());
            })
        }),
        new_session: Arc::new(move |_options: NewSessionOptions| {
            let log = Arc::clone(&hold_idle);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("new-session".to_string());
                SessionReplaceOutcome::proceeded()
            })
        }),
        fork: Arc::new(move |entry_id: String, options: ForkOptions| {
            let log = Arc::clone(&hold_fork);
            *entry_read
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(entry_id);
            *position_read
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = options.position;
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("fork".to_string());
                SessionReplaceOutcome { cancelled: true }
            })
        }),
        navigate_tree: Arc::new(move |_target_id: String, _options: NavigateTreeOptions| {
            let log = Arc::clone(&hold_nav);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("navigate".to_string());
                SessionReplaceOutcome::proceeded()
            })
        }),
        switch_session: Arc::new(
            move |_session_path: String, _options: SwitchSessionOptions| {
                let log = Arc::clone(&hold_switch);
                Box::pin(async move {
                    log.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push("switch".to_string());
                    SessionReplaceOutcome::proceeded()
                })
            },
        ),
        reload: Arc::new(move || {
            let log = Arc::clone(&hold_reload);
            Box::pin(async move {
                log.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("reload".to_string());
            })
        }),
    }));

    let context = runner.command_context();
    let outcome = context
        .new_session(NewSessionOptions {
            parent_session: Some("/parent".to_string()),
            ..NewSessionOptions::default()
        })
        .await
        .expect("the bound replacement");
    assert!(!outcome.cancelled);
    context.wait_for_idle().await.expect("the bound idle wait");
    let outcome = context
        .fork(
            "entry-1".to_string(),
            ForkOptions {
                position: Some(ForkPosition::Before),
                ..ForkOptions::default()
            },
        )
        .await
        .expect("the bound fork");
    assert!(outcome.cancelled);
    context
        .navigate_tree("target-1".to_string(), NavigateTreeOptions::default())
        .await
        .expect("the bound navigation");
    context
        .switch_session("/session.json".to_string(), SwitchSessionOptions::default())
        .await
        .expect("the bound switch");
    context.reload().await.expect("the bound reload");
    assert_eq!(
        *fork_entry
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Some("entry-1".to_string())
    );
    assert_eq!(
        *position_read2
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        Some(ForkPosition::Before)
    );
    assert_eq!(
        *command_log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec![
            "new-session".to_string(),
            "idle".to_string(),
            "fork".to_string(),
            "navigate".to_string(),
            "switch".to_string(),
            "reload".to_string()
        ]
    );

    // Unbinding resets to the no-op defaults.
    runner.bind_command_context(None);
    let context = runner.command_context();
    assert!(
        !context
            .new_session(NewSessionOptions::default())
            .await
            .expect("the reset replacement")
            .cancelled
    );
}

#[tokio::test]
async fn invalidate_first_message_wins_and_fails_the_runner_reads() {
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    let runner = runner_with_runtime(&[extension_with(Vec::new())], runtime.clone()).await;
    assert!(runner.active_tools().is_ok());

    runner.invalidate(Some("first replacement".to_string()));
    runner.invalidate(Some("second replacement".to_string()));
    assert_eq!(
        runner
            .active_tools()
            .expect_err("the invalidated runner read fails"),
        "first replacement".to_string()
    );

    // The loader runtime's staleness carries the same message so the
    // extension APIs fail closed too.
    let api = crate::extensions::api::ExtensionApi::new(
        Arc::new(Extension::new(
            "<inline:stale-check>".to_string(),
            "<inline:stale-check>".to_string(),
            false,
            source_info(),
        )),
        runtime,
        crate::core::event_bus::EventBus::default(),
        "/tmp/project".to_string(),
    );
    assert!(
        api.assert_active()
            .expect_err("the stale runtime fails the API reads")
            .contains("first replacement")
    );

    // The none-call restates upstream's guidance sentence.
    let runtime = crate::extensions::api::ExtensionRuntime::new();
    let runner = runner_with_runtime(&[extension_with(Vec::new())], runtime.clone()).await;
    runner.invalidate(None);
    let message = runner
        .active_tools()
        .expect_err("the invalidated runner read fails");
    assert!(
        message.contains("session replacement or reload"),
        "{message}"
    );
}

#[tokio::test]
async fn rebinds_land_on_the_contexts_and_the_bound_shutdown() {
    let shut: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let hold = Arc::clone(&shut);
    let actions = Arc::new(ContextActions {
        shutdown: Some(Arc::new(move || {
            *hold
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        })),
        ..ContextActions::default()
    });
    let runner = runner_with(&[extension_with(Vec::new())], actions).await;
    runner.shutdown();
    assert!(
        *shut
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        "shutdown runs the bound actions"
    );
}

#[tokio::test]
async fn session_rebindings_land_on_the_next_context() {
    let mut runner = runner_with(&[extension_with(Vec::new())], plain_actions()).await;
    runner.set_cwd("/elsewhere".to_string());
    runner.set_mode(ExtensionMode::Rpc);
    let context = runner.command_context();
    assert_eq!(
        context.cwd().expect("the rebound cwd"),
        "/elsewhere".to_string()
    );
    assert_eq!(
        context.mode().expect("the rebound mode"),
        ExtensionMode::Rpc
    );
}
