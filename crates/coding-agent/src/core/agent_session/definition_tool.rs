//! Adapt a `ToolDefinition` onto the plain `Agent`'s erased tool contract.

use std::sync::Arc;

use pi_agent_core::types::AgentTool;
use pi_agent_core::types::AgentToolResult;
use pi_ai::types::Model;

use crate::extensions::types::ExtensionContext;
use crate::extensions::types::ToolDefinition;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    pub(crate) fn definition_to_agent_tool(&self, definition: &ToolDefinition) -> AgentTool {
        let definition = Arc::new(definition.clone());
        let weak = self.self_weak();
        let execute_definition = Arc::clone(&definition);
        AgentTool {
            tool: pi_ai::types::Tool {
                name: definition.name.clone(),
                description: definition.description.clone(),
                parameters: definition.parameters.clone(),
                constrained_sampling: definition.constrained_sampling.clone(),
            },
            label: definition.label.clone(),
            prepare_arguments: definition.prepare_arguments.clone(),
            replay: None,
            execution_mode: definition.execution_mode,
            execute: Arc::new(move |id, args, token, on_update| {
                let definition = Arc::clone(&execute_definition);
                let weak = weak.clone();
                Box::pin(async move {
                    let Some(session) = weak.upgrade() else {
                        return Err(pi_agent_core::types::AgentToolError::from(
                            std::io::Error::other("the session was dropped"),
                        ));
                    };
                    let context = SessionToolContext::from_session(&session);
                    let active_before = session.active_tool_names();
                    let (signal, done) = super::helpers::bridge_signal(token);
                    let adapter = |result: &AgentToolResult,
                                   _options: Option<
                        pi_agent_core::harness::types::AgentHarnessToolUpdateOptions,
                    >| {
                        if let Some(callback) = on_update {
                            callback(result);
                        }
                    };
                    let result = (definition.execute)(
                        id,
                        args,
                        signal.as_ref(),
                        Some(&adapter),
                        Some(&context),
                    )
                    .await;
                    if let Some(done) = done {
                        done.cancel();
                    }
                    let mut result = result?;
                    let active_after = session.active_tool_names();
                    if active_before.iter().all(|name| active_after.contains(name)) {
                        let added: Vec<String> = active_after
                            .into_iter()
                            .filter(|name| !active_before.contains(name))
                            .collect();
                        if !added.is_empty() {
                            let mut merged = result.added_tool_names.clone().unwrap_or_default();
                            for name in added {
                                if !merged.contains(&name) {
                                    merged.push(name);
                                }
                            }
                            result.added_tool_names = Some(merged);
                        }
                    }
                    Ok(result)
                })
            }),
        }
    }
}

/// The context the built-in tools read during execution.
struct SessionToolContext {
    cwd: String,
    model: Option<Model>,
    thinking_level: Option<pi_agent_core::types::ThinkingLevel>,
    session_id: Option<String>,
    session_file: Option<String>,
}

impl SessionToolContext {
    fn from_session<S: SettingsStorage + 'static>(session: &super::AgentSession<S>) -> Self {
        Self {
            cwd: session.cwd.clone(),
            model: session.model(),
            thinking_level: Some(session.thinking_level()),
            session_id: Some(session.session_id()),
            session_file: session.session_file(),
        }
    }
}

impl ExtensionContext for SessionToolContext {
    fn cwd(&self) -> &str {
        &self.cwd
    }

    fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    fn thinking_level(&self) -> Option<pi_agent_core::types::ThinkingLevel> {
        self.thinking_level
    }

    fn session_id(&self) -> Option<String> {
        self.session_id.clone()
    }

    fn session_file(&self) -> Option<String> {
        self.session_file.clone()
    }
}
