//! Fixtures shared by the `ext_reg_*` suites, porting the inline helpers
//! upstream's regression harness defines at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

use std::sync::Arc;
use std::sync::Mutex;

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::AgentTool;
use pi_agent_core::types::AgentToolResult;
use pi_agent_core::types::StreamFn;
use pi_ai::types::AssistantBlock;
use pi_ai::types::AssistantMessage;
use pi_ai::types::AssistantMessageEvent;
use pi_ai::types::Message;
use pi_ai::types::StopReason;
use pi_ai::types::TextContent;
use pi_ai::types::ToolResultBlock;
use pi_ai::utils::event_stream::assistant_message_event_stream;
use pi_coding_agent::core::agent_session::AgentSession;
use pi_coding_agent::extensions::api::ExtensionApi;
use pi_coding_agent::extensions::api::InlineExtension;
use pi_coding_agent::extensions::events::ExtensionEvent;
use pi_coding_agent::extensions::results::HandlerResult;
use pi_coding_agent::extensions::types::ToolDefinition;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::InMemorySettingsStorage;
use serde_json::json;

/// The suite's session handle.
pub type Session = Arc<AgentSession<InMemorySettingsStorage>>;

/// The slot a factory publishes its captured `pi` into.
pub type ApiSlot = Arc<Mutex<Option<Arc<ExtensionApi>>>>;

/// The provider/tool names a scripted stream observed per request.
pub type NameLog = Arc<Mutex<Vec<Vec<String>>>>;

/// One in-memory session manager rooted at `root`.
///
/// # Panics
/// The manager's own construction failure, which the fixture cannot
/// satisfy.
#[must_use]
pub fn manager(root: &str) -> SessionManager {
    SessionManager::in_memory(Some(root), None, None).expect("session")
}

/// A successful text tool result.
#[must_use]
pub fn ok_result(text: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        })],
        details: serde_json::Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// The wire role name of one agent message.
#[must_use]
pub const fn role(message: &AgentMessage) -> &'static str {
    match message {
        AgentMessage::Standard(Message::User(_)) => "user",
        AgentMessage::Standard(Message::Assistant(_)) => "assistant",
        AgentMessage::Standard(Message::ToolResult(_)) => "toolResult",
        AgentMessage::Custom(_) => "custom",
    }
}

/// Every transcript role in order.
#[must_use]
pub fn roles(session: &Session) -> Vec<String> {
    session
        .agent()
        .state()
        .messages
        .iter()
        .map(|message| role(message).to_owned())
        .collect()
}

/// The transcript's user texts, upstream's `getUserTexts`.
#[must_use]
pub fn user_texts(session: &Session) -> Vec<String> {
    session
        .agent()
        .state()
        .messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(Message::User(user)) => {
                Some(pi_ai::utils::text::content_text_with(&user.content, ""))
            }
            _ => None,
        })
        .collect()
}

/// The transcript's assistant texts, upstream's `getAssistantTexts`.
#[must_use]
pub fn assistant_texts(session: &Session) -> Vec<String> {
    session
        .agent()
        .state()
        .messages
        .iter()
        .filter_map(|message| match message {
            AgentMessage::Standard(Message::Assistant(assistant)) => Some(
                assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantBlock::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .collect()
}

/// A tool result's text content.
#[must_use]
pub fn tool_result_text(message: &Message) -> String {
    match message {
        Message::ToolResult(result) => result
            .content
            .iter()
            .filter_map(|block| match block {
                ToolResultBlock::Text(text) => Some(text.text.clone()),
                ToolResultBlock::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A stream that answers one scripted message per request, clamping at the
/// last entry like upstream's faux queue.
#[must_use]
pub fn scripted(messages: Vec<AssistantMessage>) -> StreamFn {
    let index = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    Arc::new(move |_model, _context, options| {
        let stream = assistant_message_event_stream();
        // The faux provider honours the request's cancellation, upstream's
        // `AbortSignal` check; without it an aborted run would replay the
        // script forever.
        if let Some(token) = options.and_then(|options| options.transport_options.signal.clone())
            && token.is_cancelled()
        {
            let mut message = super::agent_session::assistant_message("", 0);
            message.stop_reason = StopReason::Aborted;
            message.error_message = Some("Aborted".to_owned());
            stream.push(AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                error: message,
            });
            return stream;
        }
        let position = index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .min(messages.len() - 1);
        let message = messages[position].clone();
        stream.push(AssistantMessageEvent::Done {
            reason: message.stop_reason,
            message,
        });
        stream
    })
}

/// An assistant message carrying tool calls, upstream's
/// `fauxAssistantMessage([...fauxToolCall])`.
#[must_use]
pub fn assistant_with_calls(
    calls: &[(&str, &str, serde_json::Value)],
    text: &str,
) -> AssistantMessage {
    let mut message = super::agent_session::assistant_message("", 0);
    message.content = calls
        .iter()
        .map(|(id, name, arguments)| super::agent_session::tool_call_block(id, name, arguments))
        .collect();
    if !text.is_empty() {
        message.content.push(AssistantBlock::Text(TextContent {
            text: text.to_owned(),
            text_signature: None,
        }));
    }
    message.stop_reason = StopReason::ToolUse;
    message
}

/// A tool that answers `respond`'s text from the call's argument.
#[must_use]
pub fn responding_tool(
    name: &str,
    respond: Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>,
) -> AgentTool {
    AgentTool {
        tool: pi_ai::types::Tool {
            name: name.to_owned(),
            description: format!("{name} tool"),
            parameters: json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
            }),
            constrained_sampling: None,
        },
        label: name.to_owned(),
        prepare_arguments: None,
        execute: Arc::new(move |_id, args, _signal, _on_update| {
            let text = respond(args);
            Box::pin(async move { Ok(ok_result(&text)) })
        }),
        replay: None,
        execution_mode: None,
    }
}

/// The suite's echo tool.
#[must_use]
pub fn echo_tool() -> AgentTool {
    responding_tool(
        "echo",
        Arc::new(|args| {
            args.get("text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        }),
    )
}

/// One extension tool definition answering `ok`.
#[must_use]
pub fn tool_definition(name: &str, description: &str, snippet: Option<&str>) -> ToolDefinition {
    ToolDefinition {
        name: name.to_owned(),
        label: name.to_owned(),
        description: description.to_owned(),
        prompt_snippet: snippet.map(str::to_owned),
        prompt_guidelines: None,
        parameters: json!({ "type": "object" }),
        constrained_sampling: None,
        render_shell: None,
        render_call: None,
        render_result: None,
        prepare_arguments: None,
        execution_mode: None,
        execute: Arc::new(|_id, _params, _signal, _on_update, _ctx| {
            Box::pin(async { Ok(ok_result("ok")) })
        }),
    }
}

/// A factory that registers `tools` from its `session_start` handler.
#[must_use]
pub fn session_start_tool_extension(tools: Vec<ToolDefinition>) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let tools = tools.clone();
            let handler_api = Arc::clone(&api);
            Box::pin(async move {
                api.on(
                    "session_start",
                    Arc::new(move |_event: ExtensionEvent, _ctx| {
                        let tools = tools.clone();
                        let api = Arc::clone(&handler_api);
                        Box::pin(async move {
                            for tool in tools {
                                api.register_tool(tool).expect("tool registers");
                            }
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

/// A factory that publishes its `pi` into `slot`.
#[must_use]
pub fn capture_api_extension(slot: ApiSlot) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&slot);
            Box::pin(async move {
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    }
}

/// The bindings the `waitForIdle` command block needs: a real idle wait
/// and no-op replacements.
#[must_use]
pub fn command_actions(
    session: &Session,
) -> pi_coding_agent::extensions::runner::CommandContextActions {
    use pi_coding_agent::extensions::runner::CommandContextActions;
    use pi_coding_agent::extensions::runner::SessionReplaceOutcome;

    let wait_session = Arc::clone(session);
    let wait_for_idle = Arc::new(move || {
        let session = Arc::clone(&wait_session);
        let future: pi_ai::types::BoxedFuture<'static, ()> =
            Box::pin(async move { session.wait_for_idle().await });
        future
    });
    let noop = Arc::new(|| -> pi_ai::types::BoxedFuture<'static, ()> { Box::pin(async {}) });
    let proceed = Arc::new(
        |_options: pi_coding_agent::extensions::runner::NewSessionOptions| -> pi_ai::types::BoxedFuture<
            'static,
            SessionReplaceOutcome,
        > { Box::pin(async { SessionReplaceOutcome::proceeded() }) },
    );
    let fork = Arc::new(
        |_id: String,
         _options: pi_coding_agent::extensions::runner::ForkOptions|
         -> pi_ai::types::BoxedFuture<'static, SessionReplaceOutcome> {
            Box::pin(async { SessionReplaceOutcome::proceeded() })
        },
    );
    let navigate = Arc::new(
        |_id: String,
         _options: pi_coding_agent::extensions::runner::NavigateTreeOptions|
         -> pi_ai::types::BoxedFuture<'static, SessionReplaceOutcome> {
            Box::pin(async { SessionReplaceOutcome::proceeded() })
        },
    );
    let switch = Arc::new(
        |_path: String,
         _options: pi_coding_agent::extensions::runner::SwitchSessionOptions|
         -> pi_ai::types::BoxedFuture<'static, SessionReplaceOutcome> {
            Box::pin(async { SessionReplaceOutcome::proceeded() })
        },
    );
    CommandContextActions {
        wait_for_idle,
        new_session: proceed,
        fork,
        navigate_tree: navigate,
        switch_session: switch,
        reload: noop,
    }
}

/// Run the SDK factory over one inline extension and bind it, upstream's
/// `createHarness` tool-selection options.
///
/// # Panics
/// The fixture's own setup failures, which the tests cannot satisfy.
pub async fn sdk_session(
    root: &str,
    extension: InlineExtension,
    tools: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    no_tools: Option<pi_coding_agent::core::sdk::NoTools>,
) -> Session {
    use pi_coding_agent::core::sdk::CreateAgentSessionOptions;
    use pi_coding_agent::core::sdk::create_agent_session;
    use pi_coding_agent::resource_loader::DefaultResourceLoader;
    use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
    use pi_coding_agent::settings_manager::Settings;
    use pi_coding_agent::settings_manager::SettingsManager;
    use pi_coding_agent::settings_manager::SettingsManagerCreateOptions;

    let agent_dir = format!("{root}/agent");
    std::fs::create_dir_all(&agent_dir).expect("mkdir");
    let registry = super::model_layer::create_in_memory_model_registry(
        super::model_layer::empty_auth_storage(),
    )
    .await;
    let settings_manager =
        SettingsManager::in_memory(&Settings::new(), SettingsManagerCreateOptions::default());
    let mut loader = DefaultResourceLoader::new(
        DefaultResourceLoaderOptions {
            cwd: root.to_owned(),
            agent_dir: agent_dir.clone(),
            extension_factories: vec![extension],
            no_skills: true,
            no_prompt_templates: true,
            no_themes: true,
            no_context_files: true,
            ..DefaultResourceLoaderOptions::default()
        },
        settings_manager,
    );
    loader.reload(None).await;
    let result = create_agent_session::<InMemorySettingsStorage>(CreateAgentSessionOptions {
        cwd: Some(root.to_owned()),
        agent_dir: Some(agent_dir),
        model_runtime: Some(registry.runtime().clone()),
        model: Some(super::agent_session::anthropic_model()),
        thinking_level: None,
        scoped_models: None,
        no_tools,
        tools,
        exclude_tools: exclude,
        custom_tools: None,
        resource_loader: Some(pi_coding_agent::resource_loader::shared_resource_loader(
            loader,
        )),
        session_manager: Some(manager(root)),
        settings_manager: None,
        session_start_event: None,
    })
    .await
    .expect("session creates");
    result
        .session
        .bind_extensions(pi_coding_agent::core::agent_session::ExtensionBindings::default())
        .await
        .expect("bind");
    result.session
}

/// The #6162 fixture: `switch_tools` swaps the active set to
/// `after_switch` when it executes.
#[must_use]
pub fn switching_tools_extension(slot: ApiSlot) -> InlineExtension {
    InlineExtension {
        name: None,
        hidden: false,
        factory: Arc::new(move |api: Arc<ExtensionApi>| {
            let slot = Arc::clone(&slot);
            let switch_api = Arc::clone(&api);
            Box::pin(async move {
                let mut switch_tool = tool_definition(
                    "switch_tools",
                    "Switch the active extension tool set",
                    Some("Switch to the next extension tool"),
                );
                switch_tool.execute = Arc::new(move |_id, _params, _signal, _on_update, _ctx| {
                    let api = Arc::clone(&switch_api);
                    Box::pin(async move {
                        api.set_active_tools(vec!["after_switch".to_owned()])
                            .expect("tools switch");
                        Ok(ok_result("switched"))
                    })
                });
                api.register_tool(switch_tool).expect("switch tool");
                api.register_tool(tool_definition(
                    "after_switch",
                    "Tool that should be available after switching",
                    Some("Run after the active tool set changes"),
                ))
                .expect("after switch tool");
                *slot.lock().expect("lock") = Some(api);
                Ok(())
            })
        }),
    }
}

/// Every exposed tool name, sorted.
#[must_use]
pub fn sorted_tool_names(session: &Session) -> Vec<String> {
    let mut names: Vec<String> = session
        .get_all_tools()
        .into_iter()
        .map(|tool| tool.name)
        .collect();
    names.sort();
    names
}
