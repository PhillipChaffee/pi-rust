//! Shared fixtures for the `agent-session-*` suites: the anthropic test
//! model, message constructors, and the in-memory session builder.

use std::sync::Arc;

use pi_agent_core::Agent;
use pi_agent_core::AgentInitialState;
use pi_agent_core::AgentOptions;
use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::StreamFn;
use pi_ai::types::{
    Api, AssistantBlock, AssistantMessage, AssistantMessageEvent, Message, ProviderId, StopReason,
    TextContent, Usage, UsageCost, UserContent, UserMessage,
};
use pi_ai::utils::event_stream::{AssistantMessageEventStream, assistant_message_event_stream};
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::core::agent_session::AgentSessionConfig;
use pi_coding_agent::resource_loader::DefaultResourceLoader;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use pi_coding_agent::settings_manager::Settings;
use pi_coding_agent::settings_manager::SettingsManager;
use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

/// The suite's model, upstream's `getModel("anthropic", "claude-sonnet-4-5")`.
///
/// # Panics
/// A missing builtin catalog entry, which the pinned catalog cannot produce.
#[must_use]
pub fn anthropic_model() -> pi_ai::types::Model {
    pi_ai::providers::catalog::get_builtin_model("anthropic", "claude-sonnet-4-5")
        .expect("the pinned catalog carries claude-sonnet-4-5")
}

/// A zero-cost usage reading with `input`/`output` set and the total summed.
#[must_use]
pub fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input,
        output,
        total_tokens: input + output,
        ..Usage::default()
    }
}

/// A usage reading whose cost breakdown totals `cost`.
#[must_use]
pub fn usage_with_cost(total: u64, cost: f64) -> Usage {
    Usage {
        input: total,
        total_tokens: total,
        cost: UsageCost {
            total: cost,
            ..UsageCost::default()
        },
        ..Usage::default()
    }
}

/// A user text message.
#[must_use]
pub fn user_message(text: &str, timestamp: i64) -> AgentMessage {
    AgentMessage::Standard(Message::User(UserMessage {
        content: UserContent::Text(text.to_owned()),
        timestamp,
    }))
}

/// An assistant message with the suite's zero usage and the given timestamp.
#[must_use]
pub fn assistant_message(text: &str, timestamp: i64) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        api: anthropic_model().api,
        provider: anthropic_model().provider,
        model: anthropic_model().id,
        response_model: None,
        response_id: None,
        provider_thinking_level: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        deferred: None,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp,
    }
}

/// The assistant message carrying `usage`.
#[must_use]
pub fn assistant_with_usage(text: &str, usage: Usage, timestamp: i64) -> AssistantMessage {
    AssistantMessage {
        usage,
        ..assistant_message(text, timestamp)
    }
}

/// The assistant error message carrying `error_message`.
#[must_use]
pub fn error_message(error_message: &str, timestamp: i64) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantBlock::Text(TextContent {
            text: String::new(),
            text_signature: None,
        })],
        stop_reason: StopReason::Error,
        error_message: Some(error_message.to_owned()),
        timestamp,
        ..assistant_message("", timestamp)
    }
}

/// A stream that settles with `message`.
#[must_use]
pub fn done_stream(message: AssistantMessage) -> StreamFn {
    Arc::new(move |_model, _context, _options| {
        let stream: AssistantMessageEventStream = assistant_message_event_stream();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message: message.clone(),
        });
        stream
    })
}

/// A stream that settles with a text assistant message.
#[must_use]
pub fn text_stream(text: &str) -> StreamFn {
    done_stream(assistant_message(text, 0))
}

/// One in-memory session build.
pub struct SessionSetup {
    /// The temp root the fixture's cwd and agent dir live under.
    pub root: String,
    /// The initial settings.
    pub settings: Settings,
    /// The pre-seeded manager.
    pub manager: SessionManager,
    /// The agent's initial messages.
    pub messages: Vec<AgentMessage>,
    /// The agent's stream function.
    pub stream: Option<StreamFn>,
    /// The agent's initial thinking level.
    pub thinking_level: Option<pi_agent_core::types::ThinkingLevel>,
    /// Inline extensions the loader carries.
    pub extensions: Vec<pi_coding_agent::extensions::api::InlineExtension>,
    /// The extension event bus the loader hands to every factory; a fresh
    /// one when absent.
    pub event_bus: Option<pi_coding_agent::core::event_bus::EventBus>,
    /// SDK custom tools registered outside extensions.
    pub custom_tools: Vec<pi_coding_agent::extensions::types::ToolDefinition>,
    /// Tools replacing the built-ins.
    pub base_tools_override:
        Option<std::collections::BTreeMap<String, pi_agent_core::types::AgentTool>>,
    /// Whether to emit the session-start bind after construction.
    pub bind_extensions: bool,
}

impl SessionSetup {
    /// A setup with the defaults the suites share.
    #[must_use]
    pub fn new(root: &str, manager: SessionManager) -> Self {
        Self {
            root: root.to_owned(),
            settings: Settings::new(),
            manager,
            messages: Vec::new(),
            stream: None,
            thinking_level: None,
            extensions: Vec::new(),
            event_bus: None,
            custom_tools: Vec::new(),
            base_tools_override: None,
            bind_extensions: false,
        }
    }
}

/// Build an in-memory session over a pre-seeded manager.
///
/// # Panics
/// The fixture's own setup failures, which the tests cannot satisfy.
pub async fn build(setup: SessionSetup) -> Arc<AgentSession<InMemorySettingsStorage>> {
    let SessionSetup {
        root,
        settings,
        manager,
        messages,
        stream,
        thinking_level,
        extensions,
        event_bus,
        custom_tools,
        base_tools_override,
        bind_extensions,
    } = setup;
    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let registry = super::model_layer::create_in_memory_model_registry(
        super::model_layer::empty_auth_storage(),
    )
    .await;
    let runtime = registry.runtime().clone();
    let agent = Agent::new(AgentOptions {
        initial_state: Some(AgentInitialState {
            system_prompt: "Test".to_owned(),
            model: Some(anthropic_model()),
            thinking_level,
            tools: Vec::new(),
            messages,
        }),
        stream_fn: stream,
        ..AgentOptions::default()
    });
    let settings_manager =
        SettingsManager::in_memory(&settings, SettingsManagerCreateOptions::default());
    let mut loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: root.clone(),
            agent_dir,
            extension_factories: extensions,
            event_bus,
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            ..DefaultResourceLoaderOptions::default()
        },
        settings_manager,
    );
    loader.reload(None).await;
    let session = AgentSession::new(AgentSessionConfig {
        agent,
        session_manager: manager,
        resource_loader: pi_coding_agent::resource_loader::shared_resource_loader(loader),
        cwd: root,
        scoped_models: None,
        model_runtime: runtime,
        custom_tools: Some(custom_tools),
        initial_active_tool_names: None,
        allowed_tool_names: None,
        excluded_tool_names: None,
        base_tools_override,
        extension_runner_ref: None,
        session_start_event: None,
    });
    if bind_extensions {
        session
            .bind_extensions(pi_coding_agent::core::agent_session::ExtensionBindings::default())
            .await
            .expect("extensions bind");
    }
    session
}

/// Build a session with the defaults the stats suite shares.
///
/// # Panics
/// The fixture's own setup failures, which the tests cannot satisfy.
pub async fn build_session(
    root: &str,
    settings: Settings,
    manager: SessionManager,
    messages: Vec<AgentMessage>,
    stream: Option<StreamFn>,
) -> Arc<AgentSession<InMemorySettingsStorage>> {
    let mut setup = SessionSetup::new(root, manager);
    setup.settings = settings;
    setup.messages = messages;
    setup.stream = stream;
    build(setup).await
}

/// Set the runtime API key the prompt preflight reads.
///
/// # Panics
/// The runtime's own key-set failure, which the fixture cannot satisfy.
pub async fn authorize(session: &AgentSession<InMemorySettingsStorage>) {
    session
        .model_runtime()
        .set_runtime_api_key("anthropic", "test-key".to_owned(), None)
        .await
        .expect("runtime key");
}

/// The live credential, upstream's `API_KEY` (`ANTHROPIC_OAUTH_TOKEN` or
/// `ANTHROPIC_API_KEY`).
#[must_use]
pub fn live_token() -> Option<String> {
    std::env::var("ANTHROPIC_OAUTH_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
        .or_else(|| {
            std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|token| !token.is_empty())
        })
}

/// The live anthropic stream, upstream's `streamSimple`.
#[must_use]
pub fn live_stream(token: String) -> StreamFn {
    Arc::new(move |model, context, options| {
        let mut options = options.cloned().unwrap_or_default();
        options.api_key = Some(token.clone());
        pi_ai::api::anthropic_messages::stream_simple(model, context, Some(&options))
    })
}

/// Seed the runtime credential the compaction paths resolve.
///
/// # Panics
/// The runtime's own key-set failure, which the fixture cannot satisfy.
pub async fn authorize_with(session: &AgentSession<InMemorySettingsStorage>, token: &str) {
    session
        .model_runtime()
        .set_runtime_api_key("anthropic", token.to_owned(), None)
        .await
        .expect("runtime key");
}

/// The services-backed runtime factory the suite hosts share.
pub struct RuntimeFactoryOptions {
    /// The model runtime the services carry.
    pub runtime: pi_coding_agent::model_runtime::ModelRuntime,
    /// The session model.
    pub model: pi_ai::types::Model,
    /// Builds the inline extensions each factory call carries.
    pub extensions:
        Arc<dyn Fn() -> Vec<pi_coding_agent::extensions::api::InlineExtension> + Send + Sync>,
    /// Whether the loader skips on-disk extension discovery.
    pub no_extensions: bool,
    /// The active tool names; `None` keeps the SDK default.
    pub tools: Option<Vec<String>>,
}

/// Build the runtime factory the SDK hosts share.
#[must_use]
pub fn runtime_factory(
    options: RuntimeFactoryOptions,
) -> pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeFactory<
    InMemorySettingsStorage,
> {
    use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeOptions;
    use pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeResult;
    use pi_coding_agent::core::agent_session_services::CreateAgentSessionFromServicesOptions;
    use pi_coding_agent::core::agent_session_services::CreateAgentSessionServicesOptions;
    use pi_coding_agent::core::agent_session_services::create_agent_session_from_services;
    use pi_coding_agent::core::agent_session_services::create_agent_session_services;
    use pi_coding_agent::core::sdk::CreateAgentSessionError;
    use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;

    let RuntimeFactoryOptions {
        runtime,
        model,
        extensions,
        no_extensions,
        tools,
    } = options;
    Arc::new(move |options: CreateAgentSessionRuntimeOptions| {
        let runtime = runtime.clone();
        let model = model.clone();
        let extensions = Arc::clone(&extensions);
        let tools = tools.clone();
        Box::pin(async move {
            let services = create_agent_session_services::<InMemorySettingsStorage>(
                CreateAgentSessionServicesOptions {
                    cwd: options.cwd.clone(),
                    agent_dir: Some(options.agent_dir.clone()),
                    settings_manager: None,
                    model_runtime: Some(runtime),
                    model_runtime_signal: None,
                    extension_flag_values: None,
                    resource_loader_options: Some(DefaultResourceLoaderOptions {
                        cwd: options.cwd.clone(),
                        agent_dir: options.agent_dir.clone(),
                        extension_factories: extensions(),
                        no_extensions,
                        no_skills: true,
                        no_prompt_templates: true,
                        no_themes: true,
                        no_context_files: true,
                        ..DefaultResourceLoaderOptions::default()
                    }),
                    resource_loader_reload_options: None,
                },
            )
            .await
            .map_err(CreateAgentSessionError::ModelRuntime)?;
            let services_for_result = services.clone();
            let diagnostics = services.diagnostics.clone();
            let result =
                create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
                    services,
                    session_manager: options.session_manager,
                    session_start_event: options.session_start_event,
                    model: Some(model),
                    thinking_level: None,
                    scoped_models: None,
                    tools,
                    exclude_tools: None,
                    no_tools: None,
                    custom_tools: None,
                })
                .await?;
            Ok(CreateAgentSessionRuntimeResult {
                session: result.session,
                extensions_result: result.extensions_result,
                model_fallback_message: result.model_fallback_message,
                services: services_for_result,
                diagnostics,
            })
        })
    })
}

/// A tool that counts executions and answers `respond`'s text.
#[must_use]
pub fn counting_tool(
    name: &str,
    executed: Arc<std::sync::atomic::AtomicUsize>,
    respond: Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>,
) -> pi_agent_core::types::AgentTool {
    pi_agent_core::types::AgentTool {
        tool: pi_ai::types::Tool {
            name: name.to_owned(),
            description: format!("{name} tool"),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "q": { "type": "string" }, "text": { "type": "string" } },
            }),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, args, _signal, _on_update| {
            let executed = Arc::clone(&executed);
            let respond = Arc::clone(&respond);
            let text = respond(args);
            Box::pin(async move {
                executed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(pi_agent_core::types::AgentToolResult {
                    content: vec![pi_ai::types::ToolResultBlock::Text(TextContent {
                        text,
                        text_signature: None,
                    })],
                    details: serde_json::Value::Null,
                    usage: None,
                    added_tool_names: None,
                    terminate: None,
                })
            })
        }),
        replay: None,
        execution_mode: None,
    }
}

/// Push a terminal event for `message` onto a fresh stream.
#[must_use]
pub fn terminal_stream(message: AssistantMessage) -> AssistantMessageEventStream {
    let stream = assistant_message_event_stream();
    stream.push(AssistantMessageEvent::Start {
        partial: AssistantMessage {
            content: Vec::new(),
            ..message.clone()
        },
    });
    if message.stop_reason == StopReason::Error {
        stream.push(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: message,
        });
    } else {
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
    }
    stream
}

/// A tool-call assistant block.
#[must_use]
pub fn tool_call_block(id: &str, name: &str, arguments: &serde_json::Value) -> AssistantBlock {
    AssistantBlock::ToolCall(pi_ai::types::ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments: arguments.as_object().cloned().unwrap_or_default(),
        thought_signature: None,
        namespace: None,
    })
}

/// Build the live in-memory session the credentialed suites share.
///
/// # Panics
/// The fixture's own setup failures, which the tests cannot satisfy.
pub async fn live_session(
    root: &str,
    token: &str,
    in_memory: bool,
    settings: Settings,
) -> Arc<AgentSession<InMemorySettingsStorage>> {
    std::fs::create_dir_all(root).expect("root");
    let manager = if in_memory {
        SessionManager::in_memory(Some(root), None, None).expect("session")
    } else {
        SessionManager::create(root, Some(&format!("{root}/sessions")), None).expect("session")
    };
    let mut setup = SessionSetup::new(root, manager);
    setup.settings = settings;
    setup.stream = Some(live_stream(token.to_owned()));
    let session = build(setup).await;
    authorize_with(&session, token).await;
    session
}

/// A `message_end` handler that sleeps on assistant messages, upstream's
/// delayed-handling fixture.
#[must_use]
pub fn slow_message_end_extension() -> pi_coding_agent::extensions::api::InlineExtension {
    use pi_coding_agent::extensions::api::ExtensionApi;
    use pi_coding_agent::extensions::events::ExtensionEvent;
    use pi_coding_agent::extensions::results::HandlerResult;

    pi_coding_agent::extensions::api::InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(|api: Arc<ExtensionApi>| {
            Box::pin(async move {
                api.on(
                    "message_end",
                    Arc::new(|event: ExtensionEvent, _ctx| {
                        Box::pin(async move {
                            if let ExtensionEvent::MessageEnd { message } = &event
                                && matches!(message, AgentMessage::Standard(Message::Assistant(_)))
                            {
                                tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                            }
                            HandlerResult::None
                        })
                    }),
                )
                .expect("message_end handler");
                Ok(())
            })
        }),
    }
}
