//! Bash execution recording, upstream's `executeBash` family.

use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_agent_core::types::CustomAgentMessage;

use super::helpers::now_ms;
use super::lock;
use super::types::AgentSessionEvent;
use super::types::BashExecutionOptions;
use crate::core::bash_executor::BashExecutorOptions;
use crate::core::bash_executor::BashResult;
use crate::core::bash_executor::execute_bash_with_operations;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    /// Execute a bash command and record the result.
    ///
    /// # Errors
    /// When the executor fails.
    #[expect(
        clippy::type_complexity,
        reason = "the optional chunk callback mirrors upstream's executeBash parameter"
    )]
    pub async fn execute_bash(
        &self,
        command: &str,
        on_chunk: Option<Arc<dyn Fn(&str) + Send + Sync>>,
        options: Option<BashExecutionOptions>,
    ) -> Result<BashResult, String> {
        let options = options.unwrap_or_default();
        let (_ctx, controller) = pi_agent_core::harness::context::with_cancel(
            &pi_agent_core::harness::context::background_context(),
        );
        let controller = Arc::new(controller);
        lock(&self.bash_abort_controllers).push(Arc::clone(&controller));
        let (prefix, shell_path) = {
            let settings = lock(&self.settings_manager);
            (
                settings.get_shell_command_prefix(),
                settings.get_shell_path(),
            )
        };
        let resolved = prefix.map_or_else(
            || command.to_owned(),
            |prefix| format!("{prefix}\n{command}"),
        );
        let operations = options.operations.clone().unwrap_or_else(|| {
            crate::tools::bash::create_local_bash_operations(shell_path.as_deref())
        });
        let weak = self.self_weak();
        let id = options.id.clone();
        let listener: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(move |delta: &str| {
            if let Some(callback) = &on_chunk {
                callback(delta);
            }
            if let Some(session) = weak.upgrade() {
                session.emit(AgentSessionEvent::BashExecutionUpdate {
                    id: id.clone(),
                    delta: delta.to_owned(),
                });
            }
        });
        let cwd = lock(&self.session_manager).cwd().to_owned();
        let result = execute_bash_with_operations(
            &resolved,
            &cwd,
            &operations,
            Some(BashExecutorOptions {
                on_chunk: Some(listener),
                signal: Some(controller.signal().clone()),
            }),
        )
        .await
        .map_err(|error| error.to_string());
        lock(&self.bash_abort_controllers).retain(|item| !Arc::ptr_eq(item, &controller));
        let result = result?;
        self.record_bash_result(command, &result, options.exclude_from_context);
        Ok(result)
    }

    /// Record a bash result in the transcript.
    pub fn record_bash_result(
        &self,
        command: &str,
        result: &BashResult,
        exclude_from_context: bool,
    ) {
        let message = bash_execution_message(command, result, exclude_from_context);
        if self.is_streaming() {
            lock(&self.pending_bash_messages).push(message);
        } else {
            self.agent.state().messages.push(message.clone());
            let _ = lock(&self.session_manager).append_message(message);
        }
    }

    /// Cancel a running bash command.
    pub fn abort_bash(&self) {
        for controller in lock(&self.bash_abort_controllers).iter() {
            controller.abort_without_reason();
        }
    }

    /// Whether a bash command is running.
    #[must_use]
    pub fn is_bash_running(&self) -> bool {
        !lock(&self.bash_abort_controllers).is_empty()
    }

    /// Whether pending bash messages wait to flush.
    #[must_use]
    pub fn has_pending_bash_messages(&self) -> bool {
        !lock(&self.pending_bash_messages).is_empty()
    }

    pub(crate) fn flush_pending_bash_messages(&self) {
        let pending: Vec<_> = lock(&self.pending_bash_messages).drain(..).collect();
        for message in pending {
            self.agent.state().messages.push(message.clone());
            let _ = lock(&self.session_manager).append_message(message);
        }
    }
}

fn bash_execution_message(
    command: &str,
    result: &BashResult,
    exclude_from_context: bool,
) -> AgentMessage {
    let mut data = serde_json::Map::new();
    data.insert(
        "command".to_owned(),
        serde_json::Value::String(command.to_owned()),
    );
    data.insert(
        "output".to_owned(),
        serde_json::Value::String(result.output.clone()),
    );
    if let Some(exit_code) = result.exit_code {
        data.insert("exitCode".to_owned(), serde_json::Value::from(exit_code));
    }
    data.insert(
        "cancelled".to_owned(),
        serde_json::Value::Bool(result.cancelled),
    );
    data.insert(
        "truncated".to_owned(),
        serde_json::Value::Bool(result.truncated),
    );
    if let Some(path) = &result.full_output_path {
        data.insert(
            "fullOutputPath".to_owned(),
            serde_json::Value::String(path.clone()),
        );
    }
    if exclude_from_context {
        data.insert(
            "excludeFromContext".to_owned(),
            serde_json::Value::Bool(true),
        );
    }
    AgentMessage::Custom(CustomAgentMessage {
        role: "bashExecution".to_owned(),
        timestamp: now_ms(),
        data,
    })
}
