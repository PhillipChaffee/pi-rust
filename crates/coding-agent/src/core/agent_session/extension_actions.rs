//! The message/entry/tool action bundle the extension runner binds.

use std::sync::Arc;

use pi_ai::types::BoxedFuture;

use super::lock;
use crate::core::slash_commands::SlashCommandInfo;
use crate::core::slash_commands::SlashCommandSource;
use crate::extensions::runner::ExtensionActions;
use crate::settings_manager::SettingsStorage;

impl<S: SettingsStorage + 'static> super::AgentSession<S> {
    #[expect(
        clippy::too_many_lines,
        reason = "the action bundle mirrors upstream's single bindCore object literal"
    )]
    pub(crate) fn extension_actions(&self) -> ExtensionActions {
        let weak = self.self_weak();
        let weak_user = weak.clone();
        let weak_append = weak.clone();
        let weak_name = weak.clone();
        let weak_read_name = weak.clone();
        let weak_label = weak.clone();
        let weak_active = weak.clone();
        let weak_all = weak.clone();
        let weak_set_active = weak.clone();
        let weak_refresh = weak.clone();
        let weak_commands = weak.clone();
        let weak_model = weak.clone();
        let weak_thinking = weak.clone();
        let weak_set_thinking = weak.clone();
        ExtensionActions {
            send_message: Arc::new(move |message, options| {
                if let Some(session) = weak.upgrade() {
                    let runner = session.extension_runner();
                    tokio::spawn(async move {
                        if let Err(error) = session.send_custom_message(message, options).await {
                            runner.emit_error(&extension_error(
                                "<runtime>",
                                "send_message",
                                &error,
                            ));
                        }
                    });
                }
            }),
            send_user_message: Arc::new(move |content, options| {
                if let Some(session) = weak_user.upgrade() {
                    let runner = session.extension_runner();
                    tokio::spawn(async move {
                        if let Err(error) = session.send_user_message(content, options).await {
                            runner.emit_error(&extension_error(
                                "<runtime>",
                                "send_user_message",
                                &error,
                            ));
                        }
                    });
                }
            }),
            append_entry: Arc::new(move |custom_type, data| {
                if let Some(session) = weak_append.upgrade() {
                    let entry_id = lock(&session.session_manager)
                        .append_custom_entry(&custom_type, data)
                        .ok();
                    let entry = entry_id
                        .and_then(|id| lock(&session.session_manager).get_entry(&id).cloned());
                    if let Some(crate::session_manager::FileEntry::Entry(entry)) = entry {
                        session.emit(super::types::AgentSessionEvent::EntryAppended { entry });
                    }
                }
            }),
            set_session_name: Arc::new(move |name| {
                if let Some(session) = weak_name.upgrade() {
                    session.set_session_name(&name);
                }
            }),
            get_session_name: Arc::new(move || {
                weak_read_name
                    .upgrade()
                    .and_then(|session| session.session_name())
            }),
            set_label: Arc::new(move |entry_id, label| {
                if let Some(session) = weak_label.upgrade() {
                    let _ = lock(&session.session_manager)
                        .append_label_change(&entry_id, label.as_deref());
                }
            }),
            get_active_tools: Arc::new(move || {
                weak_active
                    .upgrade()
                    .map_or_else(Vec::new, |session| session.active_tool_names())
            }),
            get_all_tools: Arc::new(move || {
                weak_all
                    .upgrade()
                    .map_or_else(Vec::new, |session| session.get_all_tools())
            }),
            set_active_tools: Arc::new(move |names| {
                if let Some(session) = weak_set_active.upgrade() {
                    session.set_active_tools_by_name(&names);
                }
            }),
            refresh_tools: Arc::new(move || {
                if let Some(session) = weak_refresh.upgrade() {
                    session.refresh_tool_registry(None, false);
                }
            }),
            get_commands: Arc::new(move || {
                weak_commands
                    .upgrade()
                    .map_or_else(Vec::new, |session| session.command_infos())
            }),
            set_model: Arc::new(move |model| {
                let weak = weak_model.clone();
                let future: BoxedFuture<'static, bool> = Box::pin(async move {
                    let Some(session) = weak.upgrade() else {
                        return false;
                    };
                    if !session.model_runtime.has_configured_auth(&model.provider.0) {
                        return false;
                    }
                    session
                        .set_model(
                            model,
                            crate::core::agent_session::ModelMutationOptions::default(),
                        )
                        .await
                        .is_ok()
                });
                future
            }),
            get_thinking_level: Arc::new(move || {
                weak_thinking
                    .upgrade()
                    .map_or(pi_agent_core::types::ThinkingLevel::Off, |session| {
                        session.thinking_level()
                    })
            }),
            set_thinking_level: Arc::new(move |level| {
                if let Some(session) = weak_set_thinking.upgrade() {
                    session.set_thinking_level(
                        level,
                        crate::core::agent_session::ModelMutationOptions::default(),
                    );
                }
            }),
        }
    }

    pub(crate) fn command_infos(&self) -> Vec<SlashCommandInfo> {
        let runner = self.extension_runner();
        let mut commands: Vec<SlashCommandInfo> = runner
            .get_registered_commands()
            .into_iter()
            .map(|command| SlashCommandInfo {
                name: command.invocation_name,
                description: command.description,
                source: SlashCommandSource::Extension,
                source_info: command.source_info,
            })
            .collect();
        commands.extend(
            self.prompt_templates()
                .into_iter()
                .map(|template| SlashCommandInfo {
                    name: template.name,
                    description: Some(template.description),
                    source: SlashCommandSource::Prompt,
                    source_info: template.source_info,
                }),
        );
        let skills = self.with_loader(|loader| loader.get_skills().skills);
        commands.extend(skills.into_iter().map(|skill| SlashCommandInfo {
            name: format!("skill:{}", skill.name),
            description: Some(skill.description),
            source: SlashCommandSource::Skill,
            source_info: skill.source_info,
        }));
        commands
    }
}

fn extension_error(
    path: &str,
    event: &str,
    error: &str,
) -> crate::extensions::runtime::ExtensionError {
    crate::extensions::runtime::ExtensionError {
        extension_path: path.to_owned(),
        event: event.to_owned(),
        error: error.to_owned(),
        stack: None,
    }
}
