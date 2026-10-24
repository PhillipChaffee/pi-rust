//! `test/agent-session-dynamic-tools.test.ts` 1:1 at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics by design"
)]

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentTool;
use pi_agent_core::types::ThinkingLevel;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::types::ToolDefinition;
use pi_coding_agent::session_manager::NewSessionOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::tools::bash::BashSpawnContext;
use pi_coding_agent::tools::bash::BashSpawnHook;
use pi_coding_agent::tools::bash::BashToolOptions;
use pi_coding_agent::tools::bash::create_bash_tool_definition;

use common::agent_session::SessionSetup;
use common::agent_session::authorize;
use common::agent_session::build;

type Session = Arc<AgentSession<InMemorySettingsStorage>>;
type EnvCapture = Arc<Mutex<Option<BTreeMap<String, String>>>>;

fn in_memory_manager(root: &str) -> SessionManager {
    SessionManager::in_memory(Some(root), None, None).expect("session")
}

fn agent_tool(session: &Session, name: &str) -> AgentTool {
    session
        .agent()
        .state()
        .tools
        .iter()
        .find(|tool| tool.tool.name == name)
        .cloned()
        .expect("the tool is active")
}

async fn execute(session: &Session, name: &str, arguments: serde_json::Value) {
    let tool = agent_tool(session, name);
    (tool.execute)("call", &arguments, None, None)
        .await
        .expect("the tool runs");
}

fn capture_hook(capture: EnvCapture) -> BashSpawnHook {
    Arc::new(move |context: BashSpawnContext| {
        *capture.lock().expect("lock") = Some(context.env.clone());
        context
    })
}

fn bash_tool_extension(root: &str, exposed: EnvCapture, opted_out: EnvCapture) -> InlineExtension {
    let root = root.to_owned();
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let root = root.clone();
            let exposed = Arc::clone(&exposed);
            let opted_out = Arc::clone(&opted_out);
            Box::pin(async move {
                let bash = create_bash_tool_definition(
                    &root,
                    Some(BashToolOptions {
                        spawn_hook: Some(capture_hook(exposed)),
                        ..BashToolOptions::default()
                    }),
                );
                api.register_tool(bash).expect("bash registers");
                let mut hidden = create_bash_tool_definition(
                    &root,
                    Some(BashToolOptions {
                        expose_session_environment: Some(false),
                        spawn_hook: Some(capture_hook(opted_out)),
                        ..BashToolOptions::default()
                    }),
                );
                "bash_without_session_env".clone_into(&mut hidden.name);
                "bash without session env".clone_into(&mut hidden.label);
                api.register_tool(hidden).expect("opted-out bash registers");
                Ok(())
            })
        }),
    }
}

#[tokio::test]
async fn exposes_session_state_before_custom_bash_spawn_hooks_and_supports_opting_out() {
    let root = "/tmp/pi-session-dynamic-tools-bash";
    std::fs::create_dir_all(root).expect("root");
    let exposed: EnvCapture = Arc::new(Mutex::new(None));
    let opted_out: EnvCapture = Arc::new(Mutex::new(None));
    let manager = SessionManager::create(
        root,
        Some(&format!("{root}/agent/sessions")),
        Some(NewSessionOptions {
            id: Some("bash-env-test".to_owned()),
            parent_session: None,
        }),
    )
    .expect("session");
    let mut setup = SessionSetup::new(root, manager);
    setup.thinking_level = Some(ThinkingLevel::High);
    setup.extensions = vec![bash_tool_extension(
        root,
        Arc::clone(&exposed),
        Arc::clone(&opted_out),
    )];
    setup.bind_extensions = true;
    let session = build(setup).await;
    authorize(&session).await;

    assert!(session.system_prompt().contains(
        "You can inspect PI_* environment variables for current model and session details."
    ));
    execute(
        &session,
        "bash",
        serde_json::json!({ "command": "printf ok" }),
    )
    .await;
    let env = exposed.lock().expect("lock").clone().expect("captured");
    assert_eq!(env.get("PI_SESSION_ID"), Some(&session.session_id()));
    assert_eq!(env.get("PI_SESSION_FILE"), session.session_file().as_ref());
    assert_eq!(
        env.get("PI_PROVIDER").map(String::as_str),
        Some("anthropic")
    );
    assert_eq!(
        env.get("PI_MODEL").map(String::as_str),
        Some(session.model().expect("model").id.as_str())
    );
    assert_eq!(
        env.get("PI_REASONING_LEVEL").map(String::as_str),
        Some("high")
    );

    execute(
        &session,
        "bash_without_session_env",
        serde_json::json!({ "command": "printf ok" }),
    )
    .await;
    let env = opted_out.lock().expect("lock").clone().expect("captured");
    for key in [
        "PI_SESSION_ID",
        "PI_SESSION_FILE",
        "PI_PROVIDER",
        "PI_MODEL",
        "PI_REASONING_LEVEL",
    ] {
        assert!(!env.contains_key(key), "{key} is absent");
    }
}

fn tool_definition(
    name: &str,
    description: &str,
    snippet: Option<&str>,
    guidelines: Option<Vec<String>>,
) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        label: name.to_owned(),
        description: description.to_owned(),
        prompt_snippet: snippet.map(str::to_owned),
        prompt_guidelines: guidelines,
        parameters: serde_json::json!({ "type": "object" }),
        constrained_sampling: None,
        render_shell: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(|_id, _params, _signal, _on_update, _ctx| {
            Box::pin(async {
                Ok(pi_agent_core::types::AgentToolResult {
                    content: vec![pi_ai::types::ToolResultBlock::Text(
                        pi_ai::types::TextContent {
                            text: "ok".to_owned(),
                            text_signature: None,
                        },
                    )],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
    }
}

fn session_start_tool_extension(tool: ToolDefinition) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let tool = tool.clone();
            let handler_api = Arc::clone(&api);
            Box::pin(async move {
                api.on(
                    "session_start",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        let tool = tool.clone();
                        let api = Arc::clone(&handler_api);
                        Box::pin(async move {
                            api.register_tool(tool).expect("tool registers");
                            HandlerResult::None
                        })
                    }),
                )
                .expect("session_start handler");
                Ok(())
            })
        }),
    }
}

#[tokio::test]
async fn refreshes_the_tool_registry_when_tools_register_after_initialization() {
    let root = "/tmp/pi-session-dynamic-tools-registry";
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.extensions = vec![session_start_tool_extension(tool_definition(
        "dynamic_tool",
        "Tool registered from session_start",
        Some("Run dynamic test behavior"),
        Some(vec![
            "Use dynamic_tool when the user asks for dynamic behavior tests.".to_owned(),
        ]),
    ))];
    let session = build(setup).await;
    authorize(&session).await;

    assert!(
        !session
            .get_all_tools()
            .iter()
            .any(|tool| tool.name == "dynamic_tool")
    );

    session
        .bind_extensions(pi_coding_agent::core::agent_session::ExtensionBindings::default())
        .await
        .expect("bind");

    let all_tools = session.get_all_tools();
    let dynamic = all_tools
        .iter()
        .find(|tool| tool.name == "dynamic_tool")
        .expect("dynamic tool");
    let read = all_tools
        .iter()
        .find(|tool| tool.name == "read")
        .expect("read tool");
    assert_eq!(
        dynamic.prompt_guidelines.as_deref(),
        Some(
            ["Use dynamic_tool when the user asks for dynamic behavior tests.".to_owned()]
                .as_slice()
        )
    );
    assert_eq!(dynamic.source_info.path, "<inline:1>");
    assert_eq!(dynamic.source_info.source, "inline");
    assert_eq!(
        dynamic.source_info.scope,
        pi_coding_agent::source_info::SourceScope::Temporary
    );
    assert_eq!(
        dynamic.source_info.origin,
        pi_coding_agent::source_info::SourceOrigin::TopLevel
    );
    assert_eq!(read.source_info.path, "<builtin:read>");
    assert_eq!(read.source_info.source, "builtin");
    assert!(
        session
            .active_tool_names()
            .iter()
            .any(|name| name == "dynamic_tool")
    );
    assert!(
        session
            .system_prompt()
            .contains("- dynamic_tool: Run dynamic test behavior")
    );
    assert!(
        session
            .system_prompt()
            .contains("- Use dynamic_tool when the user asks for dynamic behavior tests.")
    );
}

#[tokio::test]
async fn returns_source_metadata_for_sdk_custom_tools() {
    let root = "/tmp/pi-session-dynamic-tools-sdk";
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.custom_tools = vec![tool_definition(
        "sdk_tool",
        "Tool registered through createAgentSession",
        None,
        None,
    )];
    let session = build(setup).await;
    authorize(&session).await;

    let sdk = session
        .get_all_tools()
        .into_iter()
        .find(|tool| tool.name == "sdk_tool")
        .expect("sdk tool");
    assert_eq!(sdk.source_info.path, "<sdk:sdk_tool>");
    assert_eq!(sdk.source_info.source, "sdk");
    assert_eq!(
        sdk.source_info.scope,
        pi_coding_agent::source_info::SourceScope::Temporary
    );
    assert_eq!(
        sdk.source_info.origin,
        pi_coding_agent::source_info::SourceOrigin::TopLevel
    );
    assert!(
        session
            .active_tool_names()
            .iter()
            .any(|name| name == "sdk_tool")
    );
}

#[tokio::test]
async fn keeps_custom_tools_active_but_omits_them_from_available_tools_without_a_prompt_snippet() {
    let root = "/tmp/pi-session-dynamic-tools-hidden";
    let mut setup = SessionSetup::new(root, in_memory_manager(root));
    setup.extensions = vec![session_start_tool_extension(tool_definition(
        "hidden_tool",
        "Description should not appear in available tools",
        None,
        None,
    ))];
    setup.bind_extensions = true;
    let session = build(setup).await;
    authorize(&session).await;

    assert!(
        session
            .get_all_tools()
            .iter()
            .any(|tool| tool.name == "hidden_tool")
    );
    assert!(
        session
            .active_tool_names()
            .iter()
            .any(|name| name == "hidden_tool")
    );
    assert!(!session.system_prompt().contains("hidden_tool"));
    assert!(
        !session
            .system_prompt()
            .contains("Description should not appear in available tools")
    );
}
