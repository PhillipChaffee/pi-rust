//! Runtime construction, bindings, resource discovery, and the signal seam.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::lock;
use crate::extensions::events::ExtensionEvent;
use crate::extensions::reasons::SessionStartReason;
use crate::extensions::runner::ExtensionRunner;
use crate::extensions::types::ToolDefinition;
use crate::model_registry::ModelRegistry;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    #[expect(
        clippy::option_if_let_else,
        clippy::iter_kv_map,
        reason = "the two base-tool sources read clearer as the upstream if/else and the keyed map"
    )]
    pub(crate) fn build_runtime(
        &self,
        active_tool_names: Option<Vec<String>>,
        include_all_extension_tools: bool,
    ) {
        let (auto_resize_images, shell_prefix, shell_path) = {
            let settings = lock(&self.settings_manager);
            (
                settings.get_image_auto_resize(),
                settings.get_shell_command_prefix(),
                settings.get_shell_path(),
            )
        };
        let base_definitions: BTreeMap<String, ToolDefinition> =
            if let Some(overrides) = &self.base_tools_override {
                overrides
                    .values()
                    .map(|tool| (tool.tool.name.clone(), agent_tool_to_definition(tool)))
                    .collect()
            } else {
                crate::tools::index::create_all_tool_definitions(
                    &self.cwd,
                    Some(&crate::tools::index::ToolsOptions {
                        read: Some(crate::tools::read::ReadToolOptions {
                            auto_resize_images: Some(auto_resize_images),
                            operations: None,
                        }),
                        bash: Some(crate::tools::bash::BashToolOptions {
                            command_prefix: shell_prefix,
                            shell_path,
                            ..crate::tools::bash::BashToolOptions::default()
                        }),
                        ..crate::tools::index::ToolsOptions::default()
                    }),
                )
                .into_iter()
                .map(|(_, definition)| (definition.name.clone(), definition))
                .collect()
            };
        *lock(&self.base_tool_definitions) = base_definitions;

        let extensions_result =
            self.with_loader(crate::resource_loader::DefaultResourceLoader::get_extensions);
        let extensions: Vec<Arc<crate::extensions::runtime::Extension>> = extensions_result
            .extensions
            .iter()
            .filter_map(|entry| entry.record.clone())
            .collect();
        let mode = *lock(&self.extension_mode);
        let mut runner = ExtensionRunner::new(
            extensions,
            Arc::new(self.context_actions()),
            Arc::new(|| Ok(())),
            self.cwd.clone(),
            mode,
            None,
            lock(&self.session_manager).clone(),
            ModelRegistry::new(self.model_runtime.clone()),
            extensions_result.runtime,
        );
        runner.bind_core(self.extension_actions(), Some(self.provider_actions()));
        runner.bind_command_context(lock(&self.extension_command_context_actions).clone());
        runner.set_mode(mode);
        let error_listener = lock(&self.extension_error_listener).clone();
        if let Some(listener) = error_listener.clone() {
            let unsubscribe = runner.on_error(listener);
            *lock(&self.extension_error_unsubscriber) = Some(Box::new(unsubscribe));
        }
        for failure in &extensions_result.errors {
            runner.emit_error(&crate::extensions::runtime::ExtensionError {
                extension_path: failure.path.clone(),
                event: "load".to_string(),
                error: failure.error.clone(),
                stack: None,
            });
        }
        let host_context = runner.context();
        self.with_loader(|loader| loader.set_extension_bindings(host_context, error_listener));
        let runner = Arc::new(runner);
        *lock(&self.extension_runner) = Arc::clone(&runner);
        if let Some(slot) = &self.extension_runner_ref {
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::clone(&runner));
        }
        self.refresh_tool_registry(active_tool_names, include_all_extension_tools);
    }

    /// Store and apply new extension bindings.
    ///
    /// # Errors
    /// Slice errors would surface here; the current bindings never fail.
    pub async fn bind_extensions(
        &self,
        bindings: super::types::ExtensionBindings,
    ) -> Result<(), String> {
        if let Some(mode) = bindings.mode {
            *lock(&self.extension_mode) = mode;
        }
        if let Some(actions) = bindings.command_context_actions {
            *lock(&self.extension_command_context_actions) = Some(actions);
        }
        if let Some(handler) = bindings.abort_handler {
            *lock(&self.extension_abort_handler) = Some(handler);
        }
        if let Some(handler) = bindings.shutdown_handler {
            *lock(&self.extension_shutdown_handler) = Some(handler);
        }
        if let Some(listener) = bindings.on_error {
            *lock(&self.extension_error_listener) = Some(listener);
        }
        let active = self.active_tool_names();
        self.build_runtime(Some(active), true);
        let runner = self.extension_runner();
        runner
            .emit(ExtensionEvent::SessionStart {
                reason: self.session_start_event.reason,
                previous_session_file: self.session_start_event.previous_session_file.clone(),
            })
            .await;
        let reason = if self.session_start_event.reason == SessionStartReason::Reload {
            SessionStartReason::Reload
        } else {
            SessionStartReason::Startup
        };
        self.extend_resources_from_extensions(reason).await;
        Ok(())
    }

    pub(crate) async fn extend_resources_from_extensions(&self, reason: SessionStartReason) {
        let runner = self.extension_runner();
        if !runner.has_handlers("resources_discover") {
            return;
        }
        let paths = runner
            .emit_resources_discover(self.cwd.clone(), reason)
            .await;
        if paths.skill_paths.is_empty()
            && paths.prompt_paths.is_empty()
            && paths.theme_paths.is_empty()
        {
            return;
        }
        let extension_paths = crate::resource_loader::ResourceExtensionPaths {
            skill_paths: resource_paths(&paths.skill_paths),
            prompt_paths: resource_paths(&paths.prompt_paths),
            theme_paths: resource_paths(&paths.theme_paths),
        };
        if let Some(loader) = lock(&self.resource_loader).as_mut() {
            loader.extend_resources(&extension_paths);
        }
        let active = self.active_tool_names();
        let prompt = self.rebuild_system_prompt(&active);
        (*lock(&self.base_system_prompt)).clone_from(&prompt);
        self.agent.set_system_prompt(prompt);
    }

    pub(crate) fn run_signal(&self) -> Option<pi_agent_core::harness::context::AbortSignal> {
        let token = self.agent.signal()?;
        let (signal, done) = super::helpers::bridge_signal(Some(&token));
        if let Some(done) = done {
            lock(&self.signal_watchers).push(done);
        }
        signal
    }
}

fn resource_paths(
    entries: &[crate::extensions::runner::ExtensionResourcePath],
) -> Vec<crate::resource_loader::PathWithMetadata> {
    entries
        .iter()
        .map(|entry| crate::resource_loader::PathWithMetadata {
            path: entry.path.clone(),
            metadata: crate::package_manager::PathMetadata {
                source: format!(
                    "extension:{}",
                    entry.extension_path.trim_matches(['<', '>'])
                ),
                scope: crate::source_info::SourceScope::Temporary,
                origin: crate::source_info::SourceOrigin::TopLevel,
                base_dir: (!entry.extension_path.starts_with('<'))
                    .then(|| crate::utils::paths::dirname_posix(&entry.extension_path)),
            },
        })
        .collect()
}

fn agent_tool_to_definition(tool: &pi_agent_core::types::AgentTool) -> ToolDefinition {
    let tool = tool.clone();
    ToolDefinition {
        name: tool.tool.name.clone(),
        label: tool.label.clone(),
        description: tool.tool.description.clone(),
        prompt_snippet: None,
        prompt_guidelines: None,
        parameters: tool.tool.parameters.clone(),
        constrained_sampling: tool.tool.constrained_sampling.clone(),
        render_shell: None,
        render_call: None,
        render_result: None,
        prepare_arguments: tool.prepare_arguments.clone(),
        execution_mode: tool.execution_mode,
        execute: Arc::new(move |id, args, _signal, on_update, _context| {
            let tool = tool.clone();
            Box::pin(async move {
                let _ = on_update;
                (tool.execute)(id, args, None, None).await
            })
        }),
    }
}
