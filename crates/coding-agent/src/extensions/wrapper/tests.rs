#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]

use std::sync::{Arc, Mutex};

use pi_agent_core::harness::context::{Context, background_context};
use pi_agent_core::harness::types::{AgentHarnessToolInvocation, AgentHarnessToolUpdateCallback};
use pi_agent_core::types::{AgentToolContent, AgentToolError, AgentToolResult};
use serde_json::json;

use super::{wrap_registered_tool, wrap_registered_tools};
use crate::extensions::reasons::ExtensionMode;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::runtime::{AssertActiveFn, ContextActions, RegisteredTool};
use crate::extensions::types::ToolDefinition;
use crate::model_registry::ModelRegistry;
use crate::model_runtime::{CreateModelRuntimeOptions, ModelRuntime};
use crate::session_manager::SessionManager;
use crate::source_info::{SourceInfo, SourceOrigin, SourceScope, SyntheticSourceOptions};
use pi_ai::auth::credential_store::InMemoryCredentialStore;

type SharedTools = Arc<Mutex<Vec<String>>>;

/// The stable invocation identity the tools suites share.
#[derive(Debug)]
struct TestInvocation;

impl AgentHarnessToolInvocation for TestInvocation {
    fn invocation_id(&self) -> &'static str {
        "test-result"
    }

    fn operation_id(&self) -> &'static str {
        "test-operation"
    }

    fn turn_id(&self) -> &'static str {
        "test-turn"
    }

    fn get_memo(
        &self,
        _name: &str,
        _context: &Context,
    ) -> pi_ai::types::BoxedFuture<'_, Result<Option<serde_json::Value>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn set_memo(
        &self,
        _name: &str,
        _value: Option<serde_json::Value>,
        _context: &Context,
    ) -> pi_ai::types::BoxedFuture<'_, Result<(), String>> {
        Box::pin(async { Ok(()) })
    }
}

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

async fn runner_with_tools(shared: &SharedTools, active: AssertActiveFn) -> Arc<ExtensionRunner> {
    let runtime = ModelRuntime::create(CreateModelRuntimeOptions {
        credentials: Some(Arc::new(InMemoryCredentialStore::default())),
        models_path: Some(None),
        allow_model_network: false,
        refresh_on_create: Some(false),
        ..CreateModelRuntimeOptions::default()
    })
    .await
    .expect("the runtime constructs offline");
    let mut runner = ExtensionRunner::new(
        Vec::new(),
        Arc::new(ContextActions::default()),
        active,
        "/tmp/project".to_string(),
        ExtensionMode::Print,
        None,
        SessionManager::new("/tmp/project", "/tmp/sessions", None, false, None, None)
            .expect("the in-memory manager constructs"),
        ModelRegistry::new(runtime),
        crate::extensions::api::ExtensionRuntime::new(),
    );
    let reader = Arc::clone(shared);
    runner.bind_active_tools(Arc::new(move || {
        reader
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }));
    Arc::new(runner)
}

fn registered_tool(mutation: impl Fn() + Send + Sync + Clone + 'static) -> RegisteredTool {
    let mutate = Arc::new(mutation);
    RegisteredTool {
        definition: ToolDefinition {
            name: "custom".to_string(),
            label: "custom".to_string(),
            description: "test".to_string(),
            prompt_snippet: None,
            prompt_guidelines: None,
            parameters: json!({ "type": "object" }),
            constrained_sampling: None,
            render_shell: None,
            prepare_arguments: None,
            execution_mode: None,
            execute: Arc::new(
                move |_call: &str,
                      _params: &serde_json::Value,
                      _signal: Option<&pi_agent_core::harness::context::AbortSignal>,
                      _update: Option<AgentHarnessToolUpdateCallback>,
                      _context: Option<&dyn crate::extensions::ExtensionContext>| {
                    let mutate = Arc::clone(&mutate);
                    Box::pin(async move {
                        mutate();
                        Ok::<AgentToolResult, AgentToolError>(AgentToolResult {
                            content: vec![AgentToolContent::Text(pi_ai::types::TextContent {
                                text: "ran".to_string(),
                                text_signature: None,
                            })],
                            details: serde_json::json!({}),
                            usage: None,
                            added_tool_names: None,
                            terminate: None,
                        })
                    })
                },
            ),
        },
        source_info: source_info(),
    }
}

async fn run_wrapped(
    tool: &pi_agent_core::harness::types::AgentHarnessTool,
) -> Result<AgentToolResult, AgentToolError> {
    let context = background_context();
    (tool.execute)(
        "test-call",
        &json!({}),
        None::<AgentHarnessToolUpdateCallback>,
        None,
        &TestInvocation,
        &context,
    )
    .await
}
#[tokio::test]
async fn invocation_surface_answers_stably() {
    let invocation = TestInvocation;
    assert_eq!(invocation.invocation_id(), "test-result");
    assert_eq!(invocation.operation_id(), "test-operation");
    assert_eq!(invocation.turn_id(), "test-turn");
    let context = background_context();
    let got = invocation.get_memo("any", &context).await;
    assert_eq!(got, Ok(None::<serde_json::Value>));
    invocation
        .set_memo("any", None, &context)
        .await
        .expect("the memo write is a stub");
}

#[tokio::test]
async fn wrap_registered_tool_tracks_tools_added_by_execute() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string()]));

    let runner = runner_with_tools(&state, Arc::new(|| Ok(()))).await;
    let tool = wrap_registered_tool(
        registered_tool({
            let state = Arc::clone(&state);
            move || {
                state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push("b".to_string());
            }
        }),
        Arc::clone(&runner),
    );
    let result = run_wrapped(&tool).await.expect("the tool ran");
    assert_eq!(result.added_tool_names, Some(vec!["b".to_string()]));
}

#[tokio::test]
async fn pre_carried_names_dedupe_with_the_fresh_diff() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string()]));
    let runner = runner_with_tools(&state, Arc::new(|| Ok(()))).await;
    let tool = registered_tool_that_adds_and_carries({
        let state = Arc::clone(&state);
        move || {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push("carried".to_string());
        }
    });
    let wrapped = wrap_registered_tool(tool, Arc::clone(&runner));
    let result = run_wrapped(&wrapped).await.expect("the tool ran");
    // Upstream's merge order: the result's carried names first, then
    // the fresh diff, deduped by insertion order.
    assert_eq!(
        result.added_tool_names,
        Some(vec!["fresh".to_string(), "carried".to_string()])
    );
}

fn registered_tool_that_adds_and_carries(
    mutation: impl Fn() + Send + Sync + Clone + 'static,
) -> RegisteredTool {
    let mutate = Arc::new(mutation);
    let mut tool = registered_tool(|| {});
    let mutate_fresh = Arc::clone(&mutate);
    tool.definition.execute = Arc::new(
        move |_call: &str,
              _params: &serde_json::Value,
              _signal: Option<&pi_agent_core::harness::context::AbortSignal>,
              _update: Option<AgentHarnessToolUpdateCallback>,
              _context: Option<&dyn crate::extensions::ExtensionContext>| {
            let mutate = Arc::clone(&mutate);
            let mutate_fresh = Arc::clone(&mutate_fresh);
            Box::pin(async move {
                mutate();
                Ok::<AgentToolResult, AgentToolError>(AgentToolResult {
                    content: Vec::new(),
                    details: serde_json::json!({}),
                    usage: None,
                    added_tool_names: Some({
                        mutate_fresh();
                        vec!["fresh".to_string()]
                    }),
                    terminate: None,
                })
            })
        },
    );
    tool
}

#[tokio::test]
async fn a_before_tool_disappearing_keeps_the_result_unchanged() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string(), "gone".to_string()]));
    let runner = runner_with_tools(&state, Arc::new(|| Ok(()))).await;
    let tool = wrap_registered_tool(
        registered_tool({
            let state = Arc::clone(&state);
            move || {
                state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .retain(|name| name != "gone");
            }
        }),
        Arc::clone(&runner),
    );
    let result = run_wrapped(&tool).await.expect("the tool ran");
    assert_eq!(result.added_tool_names, None);
}

#[tokio::test]
async fn no_change_and_no_names_restates_none() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string()]));
    let runner = runner_with_tools(&state, Arc::new(|| Ok(()))).await;
    let tool = wrap_registered_tool(registered_tool(|| {}), Arc::clone(&runner));
    let result = run_wrapped(&tool).await.expect("the tool ran");
    assert_eq!(result.added_tool_names, None);
}

#[tokio::test]
async fn a_stale_runtime_fails_the_call() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string()]));
    let runner = runner_with_tools(&state, Arc::new(|| Err("stale".to_string()))).await;
    let tool = wrap_registered_tool(registered_tool(|| {}), Arc::clone(&runner));
    let error = run_wrapped(&tool)
        .await
        .expect_err("the stale runtime fails");
    assert!(error.to_string().contains("stale"), "{error}");
}

#[tokio::test]
async fn wrap_registered_tools_names_follow_the_registration_order() {
    let state: SharedTools = Arc::new(Mutex::new(vec!["a".to_string()]));
    let runner = runner_with_tools(&state, Arc::new(|| Ok(()))).await;
    let tools = wrap_registered_tools(&[registered_tool(|| {}), registered_tool(|| {})], &runner);
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].tool.name, "custom");
    assert_eq!(tools[1].tool.name, "custom");
}
