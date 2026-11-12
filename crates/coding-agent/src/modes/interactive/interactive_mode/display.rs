//! Transcript rendering, status lines, notifications, loaded resources,
//! queued messages, and the extension widget/header/footer slots.
//!
//! Restatements: the loaded-resource listing keeps every section and the
//! diagnostic groups but prints one flat path line per resource instead of
//! upstream's per-scope/package tree; cache-miss notices re-derive only the
//! live message (the rebuild-time replay reads the entries index the port's
//! `collect_cache_misses` keys on, which the frame does not retain).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::AgentMessage;
use pi_ai::types::{AssistantMessage, Message, StopReason};
use pi_tui::components::loader::LoaderIndicatorOptions;
use pi_tui::components::{Markdown, Spacer, Text, TruncatedText};
use pi_tui::tui::{Component, Container, ListenerId, TuiInputListener};

use crate::cache_stats::{CACHE_TTL_MS, CacheMiss, compute_cache_waste, detect_cache_miss};
use crate::core::agent_session::SessionStats;
use crate::modes::interactive::components::assistant_message::{
    AssistantMessageComponent, AssistantMessageOptions,
};
use crate::modes::interactive::components::bash_execution::BashExecutionComponent;
use crate::modes::interactive::components::branch_summary_message::BranchSummaryMessageComponent;
use crate::modes::interactive::components::compaction_summary_message::CompactionSummaryMessageComponent;
use crate::modes::interactive::components::custom_entry::CustomEntryComponent;
use crate::modes::interactive::components::custom_message::CustomMessageComponent;
use crate::modes::interactive::components::dynamic_border::DynamicBorder;
use crate::modes::interactive::components::footer::format_tokens;
use crate::modes::interactive::components::skill_invocation_message::SkillInvocationMessageComponent;
use crate::modes::interactive::components::status_indicator::{
    StatusIndicator, StatusIndicatorKind,
};
use crate::modes::interactive::components::tool_execution::ToolExecutionComponent;
use crate::modes::interactive::components::user_message::UserMessageComponent;
use crate::modes::interactive::theme::theme::{bold, get_markdown_theme, theme};
use crate::session_manager::{FileEntry, SessionEntry, session_entry_to_context_messages};
use crate::settings_manager::SettingsStorageKind;
use crate::source_info::SourceInfo;
use crate::usage_totals::get_usage_cost_breakdown;

use super::{CompactionQueuedMessage, InteractiveMode, QueueDelivery};

/// A text row whose content is rebuilt from a collapsed/expanded source,
/// upstream's `ExpandableText`.
pub(crate) struct ExpandableText {
    text: Text,
    get_collapsed: Box<dyn Fn() -> String>,
    get_expanded: Box<dyn Fn() -> String>,
}

impl ExpandableText {
    /// Build the row in `expanded`'s state.
    pub(crate) fn new(
        get_collapsed: impl Fn() -> String + 'static,
        get_expanded: impl Fn() -> String + 'static,
        expanded: bool,
        padding_x: usize,
        padding_y: usize,
    ) -> Self {
        let initial = if expanded {
            get_expanded()
        } else {
            get_collapsed()
        };
        Self {
            text: Text::with_padding(initial, padding_x, padding_y),
            get_collapsed: Box::new(get_collapsed),
            get_expanded: Box::new(get_expanded),
        }
    }

    /// Apply the expanded state.
    pub(crate) fn apply_expanded(&self, expanded: bool) {
        let text = if expanded {
            (self.get_expanded)()
        } else {
            (self.get_collapsed)()
        };
        self.text.set_text(&text);
    }
}

impl Component for ExpandableText {
    fn render(&self, width: usize) -> Vec<String> {
        self.text.render(width)
    }
    fn invalidate(&self) {
        self.text.invalidate();
    }
}

/// Quote a path for a shell when it carries characters outside the safe set.
pub(crate) fn quote_if_needed(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-./~:@".contains(character))
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// Build an error tool result.
pub(crate) fn error_tool_result(text: &str) -> pi_agent_core::types::AgentToolResult {
    pi_agent_core::types::AgentToolResult {
        content: vec![pi_ai::types::ToolResultBlock::Text(
            pi_ai::types::TextContent {
                text: text.to_owned(),
                text_signature: None,
            },
        )],
        details: serde_json::Value::Null,
        usage: None,
        added_tool_names: None,
        terminate: None,
    }
}

/// One canonical IronCalc range reference converted to a tool result, the
/// error path's only caller.
fn error_line(message: &str) -> Rc<dyn Component> {
    Rc::new(Text::with_padding(
        theme().fg("error", &format!("Error: {message}")),
        1,
        0,
    ))
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// The assistant-row options every live and replayed row shares.
    pub(crate) fn assistant_options(&self) -> AssistantMessageOptions {
        AssistantMessageOptions {
            hide_thinking_block: self.hide_thinking_block.get(),
            markdown_theme: Some(self.markdown_theme_with_settings()),
            hidden_thinking_label: self.hidden_thinking_label.borrow().clone(),
            output_pad: self.output_pad.get(),
            markdown_transformers: self.markdown_transformers(),
        }
    }

    /// The Markdown theme with the code-block indent setting applied.
    pub(crate) fn markdown_theme_with_settings(&self) -> pi_tui::components::MarkdownTheme {
        let mut markdown = get_markdown_theme();
        let indent = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_code_block_indent();
        if !indent.is_empty() {
            markdown.code_block_indent = Some(indent);
        }
        markdown
    }

    /// The Markdown transformers the message rows apply: the mermaid
    /// transformer then the extensions' registered transformers.
    pub(crate) fn markdown_transformers(
        &self,
    ) -> Arc<Vec<crate::extensions::types::MarkdownTransformer>> {
        let extension = self.session().extension_runner();
        let mut transformers = vec![Arc::clone(&self.mermaid_markdown_transformer)];
        transformers.extend(extension.get_markdown_transformers());
        Arc::new(transformers)
    }

    /// The startup-expansion state, upstream's `getStartupExpansionState`.
    /// Repaint the editor border for the current thinking level or bash mode.
    pub(crate) fn update_editor_border_color(&self) {
        let color = if self.is_bash_mode.get() {
            crate::modes::interactive::theme::theme::get_bash_mode_border_color()
        } else {
            crate::modes::interactive::theme::theme::get_thinking_border_color(
                self.session().thinking_level(),
            )
        };
        let wrapped: pi_tui::components::editor::EditorColorFn =
            Rc::new(move |text: &str| color(text));
        self.default_editor.set_border_color(Rc::clone(&wrapped));
        self.editor().set_border_color(wrapped);
        self.tui.request_render(false);
    }

    pub(crate) fn get_startup_expansion_state(&self) -> bool {
        self.options.borrow().verbose || self.tool_output_expanded.get()
    }

    /// Show an error line, upstream's `showError`.
    pub(crate) fn show_error(&self, message: &str) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(error_line(message));
        self.tui.request_render(false);
    }

    /// Show a warning line, upstream's `showWarning`.
    pub(crate) fn show_warning(&self, message: &str) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            theme().fg("warning", &format!("Warning: {message}")),
            1,
            0,
        ));
        self.chat_container.add_child(line);
        self.tui.request_render(false);
    }

    /// Show a dim status line, coalescing consecutive updates, upstream's
    /// `showStatus`.
    pub(crate) fn show_status(&self, message: &str) {
        let last = self.chat_container.children.borrow().last().cloned();
        let second_last = {
            let children = self.chat_container.children.borrow();
            let len = children.len();
            (len > 1).then(|| Rc::clone(&children[len - 2]))
        };
        let last_matches = last.as_ref().is_some_and(|last| {
            self.last_status_text
                .borrow()
                .as_ref()
                .is_some_and(|text| Rc::as_ptr(text).cast::<()>() == Rc::as_ptr(last).cast::<()>())
        });
        if last_matches
            && second_last.as_ref().is_some_and(|second| {
                self.last_status_spacer
                    .borrow()
                    .as_ref()
                    .is_some_and(|spacer| {
                        Rc::as_ptr(spacer).cast::<()>() == Rc::as_ptr(second).cast::<()>()
                    })
            })
        {
            if let Some(text) = self.last_status_text.borrow().as_ref() {
                text.set_text(&theme().fg("dim", message));
            }
            self.tui.request_render(false);
            return;
        }
        let spacer = Rc::new(Spacer::new());
        let text = Rc::new(Text::with_padding(theme().fg("dim", message), 1, 0));
        let text_child: Rc<dyn Component> = text.clone();
        self.chat_container.add_child(spacer.clone());
        self.chat_container.add_child(text_child);
        *self.last_status_spacer.borrow_mut() = Some(spacer);
        *self.last_status_text.borrow_mut() = Some(text);
        self.tui.request_render(false);
    }

    /// Show a managed-tool status update, upstream's `showManagedToolStatus`.
    pub(crate) fn show_managed_tool_status(
        &self,
        status: &crate::utils::tools_manager::ToolStatus,
    ) {
        if !self.managed_tool_status_started.get() {
            self.chat_container.add_child(Rc::new(Spacer::new()));
            self.managed_tool_status_started.set(true);
        }
        let (message, color) = match status {
            crate::utils::tools_manager::ToolStatus::Warning(message) => {
                (format!("Warning: {message}"), "warning")
            }
            crate::utils::tools_manager::ToolStatus::Info(message) => (message.clone(), "dim"),
        };
        let line: Rc<dyn Component> =
            Rc::new(Text::with_padding(theme().fg(color, &message), 1, 0));
        self.chat_container.add_child(line);
        *self.last_status_spacer.borrow_mut() = None;
        *self.last_status_text.borrow_mut() = None;
        self.tui.request_render(false);
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Add one agent message to the transcript, upstream's
    /// `addMessageToChat`.
    pub(crate) fn add_message_to_chat(&self, message: &AgentMessage, populate_history: bool) {
        match message {
            AgentMessage::Custom(custom) => self.add_custom_agent_message(custom),
            AgentMessage::Standard(Message::User(user)) => {
                let text = pi_ai::utils::text::content_text(&user.content);
                if text.is_empty() {
                    return;
                }
                if !self.chat_container.children.borrow().is_empty() {
                    self.chat_container.add_child(Rc::new(Spacer::new()));
                }
                if let Some(skill_block) = crate::core::agent_session::parse_skill_block(&text) {
                    let component = Rc::new(SkillInvocationMessageComponent::new(
                        skill_block.clone(),
                        Some(self.markdown_theme_with_settings()),
                    ));
                    component.set_expanded(self.tool_output_expanded.get());
                    self.chat_container.add_child(component);
                    if let Some(user_message) = &skill_block.user_message {
                        self.chat_container.add_child(Rc::new(Spacer::new()));
                        self.chat_container
                            .add_child(Rc::new(UserMessageComponent::new(
                                user_message,
                                Some(self.markdown_theme_with_settings()),
                                self.output_pad.get(),
                                self.markdown_transformers(),
                            )));
                    }
                } else {
                    self.chat_container
                        .add_child(Rc::new(UserMessageComponent::new(
                            &text,
                            Some(self.markdown_theme_with_settings()),
                            self.output_pad.get(),
                            self.markdown_transformers(),
                        )));
                }
                if populate_history {
                    self.editor().add_to_history(&text);
                }
            }
            AgentMessage::Standard(Message::Assistant(assistant)) => {
                let component = Rc::new(AssistantMessageComponent::new(
                    Some(assistant.clone()),
                    self.assistant_options(),
                ));
                self.assistant_rows.borrow_mut().push(Rc::clone(&component));
                self.chat_container.add_child(component);
            }
            AgentMessage::Standard(Message::ToolResult(_)) => {}
        }
    }

    fn add_custom_agent_message(&self, custom: &pi_agent_core::types::CustomAgentMessage) {
        match custom.role.as_str() {
            "bashExecution" => {
                let field = |name: &str| custom.data.get(name).cloned();
                let command = field("command")
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_default();
                let exclude = field("excludeFromContext")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false);
                let component = Rc::new(BashExecutionComponent::new(
                    &command,
                    self.render_request(),
                    exclude,
                ));
                if let Some(output) =
                    field("output").and_then(|value| value.as_str().map(str::to_owned))
                {
                    component.append_output(&output);
                }
                component.set_complete(
                    field("exitCode")
                        .and_then(|value| value.as_i64())
                        .and_then(|code| i32::try_from(code).ok()),
                    field("cancelled")
                        .and_then(|value| value.as_bool())
                        .unwrap_or(false),
                    field("truncated")
                        .and_then(|value| value.as_bool())
                        .filter(|truncated| *truncated)
                        .map(|_| None)
                        .flatten(),
                    field("fullOutputPath")
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .as_deref(),
                );
                self.chat_container.add_child(component);
            }
            "custom" => {
                if custom
                    .data
                    .get("display")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false)
                {
                    let renderer = self.session().extension_runner().get_message_renderer(
                        custom
                            .data
                            .get("customType")
                            .and_then(|value| value.as_str())
                            .unwrap_or_default(),
                    );
                    let component = Rc::new(CustomMessageComponent::new(
                        custom.clone(),
                        renderer,
                        Some(self.markdown_theme_with_settings()),
                        self.output_pad.get(),
                    ));
                    component.set_expanded(self.tool_output_expanded.get());
                    self.chat_container.add_child(component);
                }
            }
            "compactionSummary" => {
                self.chat_container.add_child(Rc::new(Spacer::new()));
                let summary = custom
                    .data
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let tokens = custom
                    .data
                    .get("tokensBefore")
                    .and_then(|value| value.as_i64())
                    .unwrap_or_default();
                let component = Rc::new(CompactionSummaryMessageComponent::new(
                    pi_agent_core::harness::messages::CompactionSummaryMessage {
                        role: custom.role.clone(),
                        summary,
                        tokens_before: tokens,
                        timestamp: custom.timestamp,
                    },
                    Some(self.markdown_theme_with_settings()),
                ));
                component.set_expanded(self.tool_output_expanded.get());
                self.chat_container.add_child(component);
            }
            "branchSummary" => {
                self.chat_container.add_child(Rc::new(Spacer::new()));
                let summary = custom
                    .data
                    .get("summary")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let component = Rc::new(BranchSummaryMessageComponent::new(
                    pi_agent_core::harness::messages::BranchSummaryMessage {
                        role: custom.role.clone(),
                        summary,
                        from_id: custom
                            .data
                            .get("fromId")
                            .and_then(|value| value.as_str().map(str::to_owned)),
                        timestamp: custom.timestamp,
                    },
                    Some(self.markdown_theme_with_settings()),
                ));
                component.set_expanded(self.tool_output_expanded.get());
                self.chat_container.add_child(component);
            }
            _ => {}
        }
    }

    /// Add a custom session entry when a renderer exists.
    pub(crate) fn add_custom_entry_to_chat(&self, custom: &SessionEntry) {
        let SessionEntry::Custom(entry) = custom else {
            return;
        };
        let Some(renderer) = self
            .session()
            .extension_runner()
            .get_entry_renderer(&entry.custom_type)
        else {
            return;
        };
        let component = Rc::new(CustomEntryComponent::new(entry.clone(), renderer));
        component.set_expanded(self.tool_output_expanded.get());
        if !component.has_content() {
            return;
        }
        if let Some(streaming) = self.streaming_component.borrow().as_ref()
            && let Some(index) = self
                .chat_container
                .children
                .borrow()
                .iter()
                .position(|child| {
                    Rc::as_ptr(child).cast::<()>() == Rc::as_ptr(streaming).cast::<()>()
                })
        {
            self.chat_container
                .children
                .borrow_mut()
                .insert(index, component);
            return;
        }
        self.chat_container.add_child(component);
    }

    /// Render typed session entries to chat, upstream's
    /// `renderSessionEntries`.
    pub(crate) fn render_session_entries(
        &self,
        entries: &[FileEntry],
        update_footer: bool,
        populate_history: bool,
    ) {
        self.pending_tools.borrow_mut().clear();
        let show_notices = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_show_cache_miss_notices();
        let mut rendered_pending: std::collections::HashMap<String, Rc<ToolExecutionComponent>> =
            std::collections::HashMap::new();
        if update_footer {
            self.footer.invalidate();
            self.update_editor_border_color();
        }
        for entry in entries {
            match entry {
                FileEntry::Entry(SessionEntry::Custom(_)) => {
                    if let FileEntry::Entry(custom) = entry {
                        self.add_custom_entry_to_chat(custom);
                    }
                    continue;
                }
                FileEntry::Entry(SessionEntry::Compaction(compaction)) => {
                    if let Some(usage) = compaction.usage {
                        self.add_compaction_cost_notice("Compaction", &usage);
                    }
                }
                FileEntry::Entry(SessionEntry::BranchSummary(branch)) => {
                    if let Some(usage) = branch.usage {
                        self.add_compaction_cost_notice("Branch summary", &usage);
                    }
                }
                _ => {}
            }
            for message in session_entry_to_context_messages(entry) {
                if let AgentMessage::Standard(Message::Assistant(assistant)) = &message {
                    self.add_message_to_chat(&message, populate_history);
                    for block in &assistant.content {
                        let pi_ai::types::AssistantBlock::ToolCall(call) = block else {
                            continue;
                        };
                        let component = Rc::new(ToolExecutionComponent::new(
                            &call.name,
                            &call.id,
                            serde_json::Value::Object(call.arguments.clone()),
                            self.tool_options(),
                            self.registered_tool_definition(&call.name),
                            self.render_request(),
                            &self.session().cwd(),
                        ));
                        component.set_expanded(self.tool_output_expanded.get());
                        self.chat_container.add_child(component.clone());
                        if matches!(
                            assistant.stop_reason,
                            StopReason::Aborted | StopReason::Error
                        ) {
                            let error = assistant
                                .error_message
                                .clone()
                                .unwrap_or_else(|| "Error".to_owned());
                            component.update_result(error_tool_result(&error), true, false);
                        } else {
                            rendered_pending.insert(call.id.clone(), component);
                        }
                    }
                    if show_notices
                        && !matches!(
                            assistant.stop_reason,
                            StopReason::Aborted | StopReason::Error
                        )
                    {
                        self.maybe_show_assistant_diagnostics(assistant);
                    }
                } else if let AgentMessage::Standard(Message::ToolResult(result)) = &message {
                    if let Some(component) = rendered_pending.remove(&result.tool_call_id) {
                        component.update_result(
                            pi_agent_core::harness::execution::tools::tool_result_from_message(
                                result, false,
                            ),
                            result.is_error,
                            false,
                        );
                    }
                } else {
                    self.add_message_to_chat(&message, populate_history);
                }
            }
        }
        self.pending_tools.borrow_mut().extend(rendered_pending);
        self.tui.request_render(false);
    }

    /// Render the initial session context, upstream's `renderInitialMessages`.
    pub(crate) fn render_initial_messages(&self) {
        let entries: Vec<FileEntry> = {
            let session = self.session();
            let manager = crate::core::agent_session::lock(&session.session_manager);
            manager
                .build_context_entries()
                .into_iter()
                .cloned()
                .collect()
        };
        self.render_session_entries(&entries, true, true);
        self.render_project_trust_warning_if_needed();
        let compaction_count = {
            let session = self.session();
            let manager = crate::core::agent_session::lock(&session.session_manager);
            manager
                .entries()
                .iter()
                .filter(|entry| matches!(entry, FileEntry::Entry(SessionEntry::Compaction(_))))
                .count()
        };
        if compaction_count > 0 {
            let times = if compaction_count == 1 {
                "1 time".to_owned()
            } else {
                format!("{compaction_count} times")
            };
            self.show_status(&format!("Session compacted {times}"));
        }
    }

    fn render_project_trust_warning_if_needed(&self) {
        let session = self.session();
        let cwd = crate::core::agent_session::lock(&session.session_manager)
            .cwd()
            .to_owned();
        let trusted = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_project_trusted();
        if trusted || !crate::trust_manager::has_trust_requiring_project_resources(&cwd) {
            return;
        }
        if !self.chat_container.children.borrow().is_empty() {
            self.chat_container.add_child(Rc::new(Spacer::new()));
        }
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            theme().fg(
                "warning",
                &format!(
                    "This project is not trusted. Project {} resources and packages are ignored. Use /trust to save a trust decision, then restart pi.",
                    crate::config::CONFIG_DIR_NAME
                ),
            ),
            1,
            0,
        ));
        self.chat_container.add_child(line);
    }

    /// Rebuild the chat from the session context.
    pub(crate) fn rebuild_chat_from_messages(&self) {
        self.chat_container.clear();
        self.render_initial_messages();
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// The tool-row options from settings.
    pub(crate) fn tool_options(
        &self,
    ) -> crate::modes::interactive::components::tool_execution::ToolExecutionOptions {
        let settings = self.settings();
        let settings = settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::modes::interactive::components::tool_execution::ToolExecutionOptions {
            show_images: Some(settings.get_show_images()),
            image_width_cells: Some(
                usize::try_from(settings.get_image_width_cells()).unwrap_or(80),
            ),
        }
    }

    /// The registered tool definition with the built-in renderers applied.
    pub(crate) fn registered_tool_definition(
        &self,
        name: &str,
    ) -> Option<crate::extensions::types::ToolDefinition> {
        crate::tools::renderers::with_built_in_renderers(
            name,
            self.session().get_tool_definition(name),
        )
    }

    /// Show an Anthropic drop notice for the assistant message.
    fn maybe_show_assistant_diagnostics(&self, message: &AssistantMessage) {
        for diagnostic in message.diagnostics.as_deref().unwrap_or_default() {
            if diagnostic.kind != "anthropic_input_transformations" {
                continue;
            }
            let Some(details) = diagnostic.details.as_ref() else {
                continue;
            };
            let Some(transformations) = details
                .get("transformations")
                .and_then(|value| value.as_array())
            else {
                continue;
            };
            let dropped: Vec<String> = transformations
                .iter()
                .filter_map(|transformation| {
                    let object = transformation.as_object()?;
                    if object.get("type").and_then(|value| value.as_str())
                        != Some("thinking_dropped")
                    {
                        return None;
                    }
                    let reason = object
                        .get("reason")
                        .and_then(|value| value.as_str())
                        .unwrap_or("unknown reason");
                    let location = object
                        .get("path")
                        .and_then(|value| value.as_str())
                        .map(|path| format!(" at {path}"))
                        .unwrap_or_default();
                    Some(format!("{reason}{location}"))
                })
                .collect();
            if dropped.is_empty() {
                continue;
            }
            let noun = if dropped.len() == 1 {
                "thinking block".to_owned()
            } else {
                format!("{} thinking blocks", dropped.len())
            };
            self.chat_container.add_child(Rc::new(Spacer::new()));
            let line: Rc<dyn Component> = Rc::new(Text::with_padding(
                theme().fg(
                    "warning",
                    &format!("Anthropic dropped {noun}: {}", dropped.join("; ")),
                ),
                1,
                0,
            ));
            self.chat_container.add_child(line);
        }
    }

    /// Add a billing notice for a compaction or branch summary.
    fn add_compaction_cost_notice(&self, label: &str, usage: &pi_ai::types::Usage) {
        let show = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_show_cache_miss_notices();
        if !show {
            return;
        }
        let tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
        let cost = if usage.cost.total >= 0.01 {
            format!(" (~${:.2})", usage.cost.total)
        } else {
            String::new()
        };
        self.chat_container.add_child(Rc::new(Spacer::new()));
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            theme().fg(
                "warning",
                &format!("{label}: {} tokens billed{cost}", format_tokens(tokens)),
            ),
            1,
            0,
        ));
        self.chat_container.add_child(line);
    }

    /// Show a cache-miss notice for a just-completed assistant message.
    pub(crate) fn maybe_show_cache_miss_notice(&self, message: &AssistantMessage) {
        let show = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_show_cache_miss_notices();
        if !show {
            return;
        }
        let entries: Vec<SessionEntry> = {
            let session = self.session();
            let manager = crate::core::agent_session::lock(&session.session_manager);
            manager
                .entries()
                .into_iter()
                .filter_map(|entry| match entry {
                    FileEntry::Entry(entry) => Some(entry.clone()),
                    _ => None,
                })
                .collect()
        };
        if let Some(miss) = detect_cache_miss(
            &entries,
            message,
            &RuntimePrices(self.session().model_runtime().clone()),
        ) {
            self.add_cache_miss_notice(&miss);
        }
    }

    fn add_cache_miss_notice(&self, miss: &CacheMiss) {
        if miss.missed_tokens < 20_000 && miss.missed_cost < 0.1 {
            return;
        }
        let cost = if miss.missed_cost >= 0.01 {
            format!(" (~${:.2})", miss.missed_cost)
        } else {
            String::new()
        };
        let re_billed = format!(
            "{} tokens re-billed{cost}",
            format_tokens(miss.missed_tokens)
        );
        let label = if miss.model_changed {
            "Cache miss after model switch".to_owned()
        } else if miss.idle_ms >= CACHE_TTL_MS {
            format!("Cache miss after {}m idle", miss.idle_ms / 60_000)
        } else {
            "Cache miss".to_owned()
        };
        self.chat_container.add_child(Rc::new(Spacer::new()));
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            theme().fg("warning", &format!("{label}: {re_billed}")),
            1,
            0,
        ));
        self.chat_container.add_child(line);
    }

    /// The session statistics line, upstream's `/session` body.
    pub(crate) fn session_info_markdown(&self) -> String {
        let stats: SessionStats = self.session().session_stats();
        let session = self.session();
        let manager = crate::core::agent_session::lock(&session.session_manager);
        let name = manager.get_session_name();
        let entries: Vec<SessionEntry> = manager
            .entries()
            .into_iter()
            .filter_map(|entry| match entry {
                FileEntry::Entry(entry) => Some(entry.clone()),
                _ => None,
            })
            .collect();
        drop(manager);
        let cache_waste = compute_cache_waste(
            &entries,
            &RuntimePrices(self.session().model_runtime().clone()),
        );
        let usage_breakdown = get_usage_cost_breakdown(&entries);
        let mut info = format!("{}\n\n", bold("Session Info"));
        if let Some(name) = name {
            info += &format!("{} {name}\n", theme().fg("dim", "Name:"));
        }
        info += &format!(
            "{} {}\n",
            theme().fg("dim", "File:"),
            stats.session_file.as_deref().unwrap_or("In-memory")
        );
        info += &format!("{} {}\n\n", theme().fg("dim", "ID:"), stats.session_id);
        info += &format!("{}\n", bold("Messages"));
        info += &format!("{} {}\n", theme().fg("dim", "Total:"), stats.total_messages);
        info += &format!("{} {}\n", theme().fg("dim", "User:"), stats.user_messages);
        info += &format!(
            "{} {}\n",
            theme().fg("dim", "Assistant:"),
            stats.assistant_messages
        );
        info += &format!(
            "{} {} calls, {} results\n\n",
            theme().fg("dim", "Tools:"),
            stats.tool_calls,
            stats.tool_results
        );
        info += &format!("{}\n", bold("Tokens"));
        let tokens = stats.tokens;
        let prompt_tokens = tokens.input + tokens.cache_read + tokens.cache_write;
        info += &format!("{} {prompt_tokens}\n", theme().fg("dim", "Input:"));
        info += &format!("{} {}\n", theme().fg("dim", "Output:"), tokens.output);
        info += &format!("{} {}\n", theme().fg("dim", "Total:"), tokens.total);
        if stats.cost > 0.0 || cache_waste.missed_tokens > 0 {
            info += &format!("\n{}\n", bold("Cost"));
            info += &format!("{} ${:.3}", theme().fg("dim", "Total:"), stats.cost);
            for entry in usage_breakdown.iter().skip(1) {
                info += &format!(
                    "\n  {} ${:.3} {}",
                    theme().fg("dim", &format!("{}:", entry.key)),
                    entry.cost,
                    theme().fg("dim", &format!("({} tokens)", format_tokens(entry.tokens)))
                );
            }
        }
        info
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Refresh the queued-message rows, upstream's
    /// `updatePendingMessagesDisplay`.
    pub(crate) fn update_pending_messages_display(&self) {
        self.pending_messages_container.clear();
        let steering = self.session().steering_messages();
        let follow_up = self.session().follow_up_messages();
        let compaction_steering: Vec<String> = self
            .compaction_queued_messages
            .borrow()
            .iter()
            .filter(|message| message.mode == QueueDelivery::Steer)
            .map(|message| message.text.clone())
            .collect();
        let compaction_follow_up: Vec<String> = self
            .compaction_queued_messages
            .borrow()
            .iter()
            .filter(|message| message.mode == QueueDelivery::FollowUp)
            .map(|message| message.text.clone())
            .collect();
        let steering: Vec<String> = steering.into_iter().chain(compaction_steering).collect();
        let follow_up: Vec<String> = follow_up.into_iter().chain(compaction_follow_up).collect();
        if steering.is_empty() && follow_up.is_empty() {
            return;
        }
        self.pending_messages_container
            .add_child(Rc::new(Spacer::new()));
        for message in steering {
            self.pending_messages_container
                .add_child(Rc::new(TruncatedText::with_padding(
                    theme().fg("dim", &format!("Steering: {message}")),
                    1,
                    0,
                )));
        }
        for message in follow_up {
            self.pending_messages_container
                .add_child(Rc::new(TruncatedText::with_padding(
                    theme().fg("dim", &format!("Follow-up: {message}")),
                    1,
                    0,
                )));
        }
        let hint = crate::modes::interactive::components::keybinding_hints::key_display_text(
            "app.message.dequeue",
        );
        self.pending_messages_container
            .add_child(Rc::new(TruncatedText::with_padding(
                theme().fg("dim", &format!("↳ {hint} to edit all queued messages")),
                1,
                0,
            )));
    }

    /// Restore every queued message to the editor, upstream's
    /// `restoreQueuedMessagesToEditor`.
    pub(crate) fn restore_queued_messages_to_editor(
        &self,
        abort: bool,
        current_text: Option<&str>,
    ) -> usize {
        let (steering, follow_up) = self.session().clear_queue();
        let mut compaction = std::mem::take(&mut *self.compaction_queued_messages.borrow_mut());
        let mut all: Vec<String> = steering;
        all.extend(follow_up);
        all.extend(compaction.drain(..).map(|message| message.text));
        if all.is_empty() {
            self.update_pending_messages_display();
            if abort {
                self.session().agent().abort();
            }
            return 0;
        }
        let queued = all.join("\n\n");
        let current = current_text.map_or_else(|| self.editor().get_text(), str::to_owned);
        let combined = [queued, current]
            .into_iter()
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        self.editor().set_text(&combined);
        self.update_pending_messages_display();
        if abort {
            self.session().agent().abort();
        }
        all.len()
    }

    /// Queue a message while compaction runs, upstream's
    /// `queueCompactionMessage`.
    pub(crate) fn queue_compaction_message(&self, text: &str, mode: QueueDelivery) {
        self.compaction_queued_messages
            .borrow_mut()
            .push(CompactionQueuedMessage {
                text: text.to_owned(),
                mode,
            });
        self.editor().add_to_history(text);
        self.editor().set_text("");
        self.update_pending_messages_display();
        self.show_status("Queued message for after compaction");
    }

    /// Drain the compaction queue, upstream's `flushCompactionQueue`.
    pub(crate) fn flush_compaction_queue(&self, will_retry: bool) {
        if self.compaction_queued_messages.borrow().is_empty() {
            return;
        }
        let queued = std::mem::take(&mut *self.compaction_queued_messages.borrow_mut());
        self.update_pending_messages_display();
        let session = self.session();
        if will_retry {
            for message in queued {
                let session = Arc::clone(&session);
                tokio::spawn(async move {
                    if Self::is_extension_command_static(&session, &message.text) {
                        let _ = session.prompt(message.text, None).await;
                    } else if message.mode == QueueDelivery::FollowUp {
                        let _ = session.follow_up(message.text, None, None).await;
                    } else {
                        let _ = session.steer(message.text, None, None).await;
                    }
                });
            }
            return;
        }
        let first_prompt = queued
            .iter()
            .position(|message| !Self::is_extension_command_static(&session, &message.text));
        let Some(first_prompt) = first_prompt else {
            for message in queued {
                let session = Arc::clone(&session);
                tokio::spawn(async move {
                    let _ = session.prompt(message.text, None).await;
                });
            }
            return;
        };
        for message in &queued[..first_prompt] {
            let session = Arc::clone(&session);
            let text = message.text.clone();
            tokio::spawn(async move {
                let _ = session.prompt(text, None).await;
            });
        }
        let prompt = queued[first_prompt].clone();
        let behavior = match prompt.mode {
            QueueDelivery::Steer => crate::extensions::reasons::StreamingBehavior::Steer,
            QueueDelivery::FollowUp => crate::extensions::reasons::StreamingBehavior::FollowUp,
        };
        let session_clone = Arc::clone(&session);
        tokio::spawn(async move {
            let options = crate::core::agent_session::PromptOptions {
                streaming_behavior: Some(behavior),
                ..crate::core::agent_session::PromptOptions::default()
            };
            let _ = session_clone.prompt(prompt.text, Some(options)).await;
        });
        for message in queued.into_iter().skip(first_prompt + 1) {
            let session = Arc::clone(&session);
            tokio::spawn(async move {
                if Self::is_extension_command_static(&session, &message.text) {
                    let _ = session.prompt(message.text, None).await;
                } else if message.mode == QueueDelivery::FollowUp {
                    let _ = session.follow_up(message.text, None, None).await;
                } else {
                    let _ = session.steer(message.text, None, None).await;
                }
            });
        }
        self.update_pending_messages_display();
    }

    fn is_extension_command_static(
        session: &Arc<crate::core::agent_session::AgentSession<S>>,
        text: &str,
    ) -> bool {
        if !text.starts_with('/') {
            return false;
        }
        let name = text
            .trim_start_matches('/')
            .split(' ')
            .next()
            .unwrap_or_default();
        session.extension_runner().get_command(name).is_some()
    }

    /// Move pending bash components into the chat, upstream's
    /// `flushPendingBashComponents`.
    pub(crate) fn flush_pending_bash_components(&self) {
        let components = std::mem::take(&mut *self.pending_bash_components.borrow_mut());
        for component in components {
            let component: Rc<dyn Component> = component;
            self.pending_messages_container.remove_child(&component);
            self.chat_container.add_child(component);
        }
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show or replace the status indicator.
    pub(crate) fn show_status_indicator(&self, indicator: Rc<StatusIndicator>) {
        if let Some(previous) = self.active_status_indicator.borrow_mut().take() {
            previous.dispose();
        }
        *self.active_status_indicator.borrow_mut() = Some(Rc::clone(&indicator));
        self.active_working_indicator_embedded.set(false);
        self.status_container.clear();
        self.default_editor.set_working_status_indicator(None);
        if self.editor().embed_working_status() {
            self.editor().set_working_status_indicator(Some(indicator));
            self.active_working_indicator_embedded.set(true);
            return;
        }
        self.status_container.add_child(indicator);
    }

    /// Clear the active indicator, optionally only when it carries `kind`.
    pub(crate) fn clear_status_indicator(&self, kind: Option<StatusIndicatorKind>) {
        if let Some(kind) = kind
            && self
                .active_status_indicator
                .borrow()
                .as_ref()
                .is_none_or(|indicator| indicator.kind() != kind)
        {
            return;
        }
        let was_embedded = self.active_working_indicator_embedded.get();
        if let Some(indicator) = self.active_status_indicator.borrow_mut().take() {
            indicator.dispose();
        }
        self.active_working_indicator_embedded.set(false);
        self.status_container.clear();
        self.default_editor.set_working_status_indicator(None);
        self.editor().set_working_status_indicator(None);
        if !was_embedded
            && self.options.borrow().tui_mode == Some(pi_tui::tui::TuiMode::Regular)
            && self.tui.get_clear_on_shrink()
        {
            let idle: Rc<dyn Component> = self.idle_status.clone();
            self.status_container.add_child(idle);
        }
    }

    /// Show the working indicator.
    pub(crate) fn show_working_status_indicator(&self) {
        let message = self
            .working_message
            .borrow()
            .clone()
            .unwrap_or_else(|| Self::DEFAULT_WORKING_MESSAGE.to_owned());
        let indicator = self.build_status(
            StatusIndicatorKind::Working,
            message,
            self.working_indicator_options.borrow().clone(),
        );
        self.show_status_indicator(indicator);
    }

    /// Show or hide the working indicator.
    pub(crate) fn set_working_visible(&self, visible: bool) {
        self.working_visible.set(visible);
        if !visible {
            self.clear_status_indicator(Some(StatusIndicatorKind::Working));
        } else if self.session().is_streaming()
            && self
                .active_status_indicator
                .borrow()
                .as_ref()
                .is_none_or(|indicator| indicator.kind() != StatusIndicatorKind::Working)
        {
            self.show_working_status_indicator();
        }
        self.tui.request_render(false);
    }

    /// Set the working indicator options.
    pub(crate) fn set_working_indicator(&self, options: Option<LoaderIndicatorOptions>) {
        *self.working_indicator_options.borrow_mut() = options.clone();
        if let Some(indicator) = self.active_status_indicator.borrow().as_ref()
            && indicator.kind() == StatusIndicatorKind::Working
        {
            indicator.set_indicator(options);
        }
        self.tui.request_render(false);
    }

    /// Set the hidden-thinking label on every assistant row.
    pub(crate) fn set_hidden_thinking_label(&self, label: Option<&str>) {
        let label = label.unwrap_or(Self::DEFAULT_HIDDEN_THINKING_LABEL);
        *self.hidden_thinking_label.borrow_mut() = label.to_owned();
        self.update_thinking_block_visibility();
        self.tui.request_render(false);
    }

    /// Apply the hide-thinking setting to every assistant row.
    pub(crate) fn update_thinking_block_visibility(&self) {
        let label = self.hidden_thinking_label.borrow().clone();
        for assistant in self.assistant_rows.borrow().iter() {
            assistant.set_hide_thinking_block(self.hide_thinking_block.get());
            assistant.set_hidden_thinking_label(&label);
        }
        if let Some(component) = self.streaming_component.borrow().as_ref() {
            component.set_hide_thinking_block(self.hide_thinking_block.get());
            component.set_hidden_thinking_label(&label);
        }
        self.tui.request_render(false);
    }

    /// Toggle thinking-block visibility.
    pub(crate) fn toggle_thinking_block_visibility(&self) {
        self.hide_thinking_block
            .set(!self.hide_thinking_block.get());
        self.settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_hide_thinking_block(self.hide_thinking_block.get());
        self.update_thinking_block_visibility();
        self.show_status(&format!(
            "Thinking blocks: {}",
            if self.hide_thinking_block.get() {
                "hidden"
            } else {
                "visible"
            }
        ));
    }

    /// Expand or collapse tool output.
    pub(crate) fn set_tools_expanded(&self, expanded: bool) {
        if expanded == self.tool_output_expanded.get() {
            return;
        }
        self.tool_output_expanded.set(expanded);
        self.show_status(&format!(
            "Tool output: {}",
            if expanded { "expanded" } else { "collapsed" }
        ));
    }

    /// Toggle tool output expansion.
    pub(crate) fn toggle_tool_output_expansion(&self) {
        self.set_tools_expanded(!self.tool_output_expanded.get());
    }

    /// Clear the editor.
    pub fn clear_editor(&self) {
        self.editor().set_text("");
        self.tui.request_render(false);
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Set or clear an extension widget, upstream's `setExtensionWidget`.
    pub(crate) fn set_extension_widget(
        &self,
        key: &str,
        lines: Option<Vec<String>>,
        placement: super::UiWidgetPlacement,
    ) {
        let target = match placement {
            super::UiWidgetPlacement::AboveEditor => &self.extension_widgets_above,
            super::UiWidgetPlacement::BelowEditor => &self.extension_widgets_below,
        };
        target.borrow_mut().remove(key);
        self.extension_widgets_above.borrow_mut().remove(key);
        self.extension_widgets_below.borrow_mut().remove(key);
        let Some(lines) = lines else {
            self.render_widgets();
            return;
        };
        let container = Rc::new(Container::default());
        for line in lines.iter().take(Self::MAX_WIDGET_LINES) {
            container.add_child(Rc::new(Text::with_padding(line.clone(), 1, 0)));
        }
        if lines.len() > Self::MAX_WIDGET_LINES {
            container.add_child(Rc::new(Text::with_padding(
                theme().fg("muted", "... (widget truncated)"),
                1,
                0,
            )));
        }
        match placement {
            super::UiWidgetPlacement::AboveEditor => {
                self.extension_widgets_above
                    .borrow_mut()
                    .insert(key.to_owned(), container);
            }
            super::UiWidgetPlacement::BelowEditor => {
                self.extension_widgets_below
                    .borrow_mut()
                    .insert(key.to_owned(), container);
            }
        }
        self.render_widgets();
    }

    /// Set or clear an extension component widget.
    pub(crate) fn set_extension_widget_component(
        &self,
        key: &str,
        component: Option<Rc<dyn Component>>,
        placement: super::UiWidgetPlacement,
    ) {
        self.extension_widgets_above.borrow_mut().remove(key);
        self.extension_widgets_below.borrow_mut().remove(key);
        if let Some(component) = component {
            match placement {
                super::UiWidgetPlacement::AboveEditor => {
                    self.extension_widgets_above
                        .borrow_mut()
                        .insert(key.to_owned(), component);
                }
                super::UiWidgetPlacement::BelowEditor => {
                    self.extension_widgets_below
                        .borrow_mut()
                        .insert(key.to_owned(), component);
                }
            }
        }
        self.render_widgets();
    }

    /// Clear every extension widget.
    pub(crate) fn clear_extension_widgets(&self) {
        self.extension_widgets_above.borrow_mut().clear();
        self.extension_widgets_below.borrow_mut().clear();
        self.render_widgets();
    }

    /// Render the widget stacks into their containers.
    pub(crate) fn render_widgets(&self) {
        self.render_widget_container(
            &self.widget_container_above,
            &self.extension_widgets_above,
            true,
            true,
        );
        self.render_widget_container(
            &self.widget_container_below,
            &self.extension_widgets_below,
            false,
            false,
        );
        self.tui.request_render(false);
    }

    fn render_widget_container(
        &self,
        container: &Rc<Container>,
        widgets: &RefCell<std::collections::BTreeMap<String, Rc<dyn Component>>>,
        spacer_when_empty: bool,
        leading_spacer: bool,
    ) {
        container.clear();
        if widgets.borrow().is_empty() {
            if spacer_when_empty {
                container.add_child(Rc::new(Spacer::new()));
            }
            return;
        }
        if leading_spacer {
            container.add_child(Rc::new(Spacer::new()));
        }
        for component in widgets.borrow().values() {
            container.add_child(Rc::clone(component));
        }
    }

    /// Set a custom footer, upstream's `setExtensionFooter`.
    pub(crate) fn set_extension_footer(&self, footer: Option<Rc<dyn Component>>) {
        self.footer_container.clear();
        match footer {
            Some(footer) => {
                *self.custom_footer.borrow_mut() = Some(Rc::clone(&footer));
                self.footer_container.add_child(footer);
            }
            None => {
                *self.custom_footer.borrow_mut() = None;
                self.footer_container
                    .add_child(Rc::clone(&self.footer) as Rc<dyn Component>);
            }
        }
        self.tui.request_render(false);
    }

    /// Set a custom header, upstream's `setExtensionHeader`.
    pub(crate) fn set_extension_header(&self, header: Option<Rc<dyn Component>>) {
        let Some(built_in) = self.built_in_header.borrow().clone() else {
            return;
        };
        let current = self
            .custom_header
            .borrow()
            .clone()
            .unwrap_or_else(|| Rc::clone(&built_in));
        let index = self
            .header_container
            .children
            .borrow()
            .iter()
            .position(|child| Rc::ptr_eq(child, &current));
        match header {
            Some(header) => {
                *self.custom_header.borrow_mut() = Some(Rc::clone(&header));
                let mut children = self.header_container.children.borrow_mut();
                match index {
                    Some(index) => children[index] = header,
                    None => children.insert(0, header),
                }
            }
            None => {
                *self.custom_header.borrow_mut() = None;
                let mut children = self.header_container.children.borrow_mut();
                if let Some(index) = index {
                    children[index] = built_in;
                }
            }
        }
        self.tui.request_render(false);
    }

    /// Register an extension terminal input listener, upstream's
    /// `addExtensionTerminalInputListener`.
    pub(crate) fn add_extension_terminal_input_listener(
        &self,
        handler: TuiInputListener,
    ) -> Rc<dyn Fn()> {
        let id = self.tui.add_input_listener(Rc::clone(&handler));
        self.extension_terminal_input_subscriptions
            .borrow_mut()
            .push((id, handler));
        let tui = Rc::clone(&self.tui);
        let subscriptions = Rc::new(());
        let _ = subscriptions;
        Rc::new(move || {
            tui.remove_input_listener(id);
        })
    }

    /// Re-bind the terminal listeners after a renderer swap.
    pub(crate) fn rebind_extension_terminal_input_listeners(&self) {
        let listeners: Vec<(ListenerId, TuiInputListener)> =
            self.extension_terminal_input_subscriptions.borrow().clone();
        for (id, _) in &listeners {
            self.tui.remove_input_listener(*id);
        }
        let mut subscriptions = self.extension_terminal_input_subscriptions.borrow_mut();
        subscriptions.clear();
        for (_, listener) in listeners {
            let id = self.tui.add_input_listener(listener.clone());
            subscriptions.push((id, listener));
        }
    }

    /// Drop every terminal input listener.
    pub(crate) fn clear_extension_terminal_input_listeners(&self) {
        for (id, _) in self.extension_terminal_input_subscriptions.borrow().iter() {
            self.tui.remove_input_listener(*id);
        }
        self.extension_terminal_input_subscriptions
            .borrow_mut()
            .clear();
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Replace the home directory with `~`, upstream's `formatDisplayPath`.
    fn format_display_path(&self, path: &str) -> String {
        let Some(home) = std::env::var_os("HOME") else {
            return path.to_owned();
        };
        let home = home.to_string_lossy();
        path.strip_prefix(home.as_ref())
            .map_or_else(|| path.to_owned(), |rest| format!("~{rest}"))
    }

    /// Get a short path relative to the package root, upstream's
    /// `getShortPath`.
    fn get_short_path(&self, full_path: &str, source_info: Option<&SourceInfo>) -> String {
        let normalized = full_path.replace('\\', "/");
        if let Some(source) = source_info {
            if let Some(captures) = npm_path_capture(&normalized, &source.source) {
                return captures;
            }
            if source.source.starts_with("git:") {
                if let Some(index) = normalized.find("/git/") {
                    let rest = &normalized[index + 5..];
                    let mut parts = rest.splitn(3, '/');
                    let _ = parts.next();
                    let _ = parts.next();
                    if let Some(tail) = parts.next() {
                        return tail.to_owned();
                    }
                }
            }
        }
        self.format_display_path(full_path)
    }

    /// The section header format, upstream's `sectionHeader`.
    fn section_header(&self, name: &str) -> String {
        theme().fg("mdHeading", &format!("[{name}]"))
    }

    /// Show the loaded-resource listing, upstream's `showLoadedResources`.
    pub(crate) fn show_loaded_resources(&self, show_diagnostics_when_quiet: bool) {
        self.loaded_resources_container.clear();
        let verbose = self.options.borrow().verbose;
        let quiet = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_quiet_startup();
        let show_listing = !quiet || verbose;
        let show_diagnostics = show_listing || show_diagnostics_when_quiet;
        if !show_listing && !show_diagnostics {
            return;
        }
        let session = self.session();
        let skills = session.with_resource_loader(|loader| loader.get_skills());
        let prompts = session.with_resource_loader(|loader| loader.get_prompts());
        let themes = session.with_resource_loader(|loader| loader.get_themes());
        let extensions_result = session.with_resource_loader(|loader| loader.get_extensions());
        let extensions: Vec<_> = extensions_result
            .extensions
            .iter()
            .filter(|extension| !extension.hidden)
            .cloned()
            .collect();
        if show_listing {
            let mut context: Vec<String> = Vec::new();
            if let Some(prompt) =
                session.with_resource_loader(|loader| loader.get_system_prompt_source())
            {
                context.push(prompt.path);
            }
            context.extend(
                session
                    .with_resource_loader(|loader| loader.get_append_system_prompt_sources())
                    .into_iter()
                    .map(|source| source.path),
            );
            context.extend(
                session
                    .with_resource_loader(|loader| loader.get_agents_files())
                    .into_iter()
                    .map(|file| file.path),
            );
            if !context.is_empty() {
                self.add_loaded_section(
                    "Context",
                    &context
                        .iter()
                        .map(|path| self.format_display_path(path))
                        .collect::<Vec<_>>(),
                );
            }
            if !skills.skills.is_empty() {
                self.add_loaded_section(
                    "Skills",
                    &skills
                        .skills
                        .iter()
                        .map(|skill| {
                            format!(
                                "{} {}",
                                skill.name,
                                theme().fg(
                                    "muted",
                                    &self
                                        .get_short_path(&skill.file_path, Some(&skill.source_info))
                                )
                            )
                        })
                        .collect::<Vec<_>>(),
                );
            }
            if !prompts.prompts.is_empty() {
                self.add_loaded_section(
                    "Prompts",
                    &prompts
                        .prompts
                        .iter()
                        .map(|prompt| format!("/{}", prompt.name))
                        .collect::<Vec<_>>(),
                );
            }
            if !extensions.is_empty() {
                self.add_loaded_section(
                    "Extensions",
                    &extensions
                        .iter()
                        .map(|extension| {
                            self.get_short_path(&extension.path, extension.source_info.as_ref())
                        })
                        .collect::<Vec<_>>(),
                );
            }
            let custom_themes: Vec<_> = themes
                .themes
                .iter()
                .filter(|handle| handle.source_path.is_some())
                .collect();
            if !custom_themes.is_empty() {
                self.add_loaded_section(
                    "Themes",
                    &custom_themes
                        .iter()
                        .map(|handle| {
                            handle
                                .name
                                .clone()
                                .unwrap_or_else(|| handle.source_path.clone().unwrap_or_default())
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        if show_diagnostics {
            let runner = self.session().extension_runner();
            let mut extension_diagnostics: Vec<crate::diagnostics::ResourceDiagnostic> =
                extensions_result
                    .errors
                    .iter()
                    .map(|error| crate::diagnostics::ResourceDiagnostic {
                        kind: crate::diagnostics::ResourceDiagnosticKind::Error,
                        message: error.error.clone(),
                        path: Some(error.path.clone()),
                        collision: None,
                    })
                    .collect();
            extension_diagnostics.extend(runner.get_command_diagnostics());
            if !extension_diagnostics.is_empty() {
                self.add_loaded_diagnostics("Extension issues", &extension_diagnostics);
            }
            if !skills.diagnostics.is_empty() {
                self.add_loaded_diagnostics("Skill conflicts", &skills.diagnostics);
            }
            if !prompts.diagnostics.is_empty() {
                self.add_loaded_diagnostics("Prompt conflicts", &prompts.diagnostics);
            }
            if !themes.diagnostics.is_empty() {
                self.add_loaded_diagnostics("Theme conflicts", &themes.diagnostics);
            }
        }
        self.tui.request_render(false);
    }

    fn add_loaded_section(&self, name: &str, lines: &[String]) {
        let body = lines
            .iter()
            .map(|line| theme().fg("dim", &format!("  {line}")))
            .collect::<Vec<_>>()
            .join("\n");
        let text: Rc<dyn Component> = Rc::new(Text::with_padding(
            format!("{}\n{body}", self.section_header(name)),
            0,
            0,
        ));
        self.loaded_resources_container.add_child(text);
        self.loaded_resources_container
            .add_child(Rc::new(Spacer::new()));
    }
}

/// Capture the path under a package's `node_modules` root.
fn npm_path_capture(path: &str, source: &str) -> Option<String> {
    if !source.starts_with("npm:") {
        return None;
    }
    let marker = "node_modules/";
    let index = path.find(marker)?;
    let rest = &path[index + marker.len()..];
    let mut parts = rest.splitn(3, '/');
    let _package = parts.next()?;
    parts.next()?;
    parts.next().map(str::to_owned)
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Append one diagnostic group to the loaded-resources area.
    pub(crate) fn add_loaded_diagnostics(
        &self,
        title: &str,
        diagnostics: &[crate::diagnostics::ResourceDiagnostic],
    ) {
        let lines = diagnostics
            .iter()
            .map(|diagnostic| {
                let path = diagnostic
                    .path
                    .as_deref()
                    .map(|path| format!("{}: ", self.format_display_path(path)))
                    .unwrap_or_default();
                theme().fg(
                    if diagnostic.kind == crate::diagnostics::ResourceDiagnosticKind::Error {
                        "error"
                    } else {
                        "warning"
                    },
                    &format!("  {path}{}", diagnostic.message),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let text: Rc<dyn Component> = Rc::new(Text::with_padding(
            format!("{}\n{lines}", theme().fg("warning", &format!("[{title}]"))),
            0,
            0,
        ));
        self.loaded_resources_container.add_child(text);
        self.loaded_resources_container
            .add_child(Rc::new(Spacer::new()));
    }

    /// Render the startup changelog notices.
    pub(crate) fn show_startup_notices_if_needed(&self) {
        if self.startup_notices_shown.get() {
            return;
        }
        self.startup_notices_shown.set(true);
        let Some(markdown) = self.changelog_markdown.borrow().clone() else {
            return;
        };
        if !self.chat_container.children.borrow().is_empty() {
            self.chat_container.add_child(Rc::new(Spacer::new()));
        }
        self.chat_container
            .add_child(Rc::new(DynamicBorder::themed()));
        if self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_collapse_changelog()
        {
            let line: Rc<dyn Component> = Rc::new(Text::with_padding(
                format!("Updated to v{}.", self.version),
                1,
                0,
            ));
            self.chat_container.add_child(line);
        } else {
            self.chat_container.add_child(Rc::new(Text::with_padding(
                bold(&theme().fg("accent", "What's New")),
                1,
                0,
            )));
            self.chat_container.add_child(Rc::new(Spacer::new()));
            self.chat_container.add_child(Rc::new(Markdown::new(
                markdown,
                1,
                0,
                self.markdown_theme_with_settings(),
            )));
            self.chat_container.add_child(Rc::new(Spacer::new()));
        }
        self.chat_container
            .add_child(Rc::new(DynamicBorder::themed()));
    }

    /// Show an extension failure in the transcript, upstream's
    /// `showExtensionError`.
    pub(crate) fn show_extension_error(
        &self,
        extension_path: &str,
        error: &str,
        stack: Option<&str>,
    ) {
        let message = format!("Extension \"{extension_path}\" error: {error}");
        let line: Rc<dyn Component> =
            Rc::new(Text::with_padding(theme().fg("error", &message), 1, 0));
        self.chat_container.add_child(line);
        if let Some(stack) = stack {
            let lines = stack
                .lines()
                .skip(1)
                .map(|line| theme().fg("dim", &format!("  {}", line.trim())))
                .collect::<Vec<_>>()
                .join("\n");
            if !lines.is_empty() {
                let text: Rc<dyn Component> = Rc::new(Text::with_padding(lines, 1, 0));
                self.chat_container.add_child(text);
            }
        }
        self.tui.request_render(false);
    }

    /// Show a new-version notification.
    pub(crate) fn show_new_version_notification(
        &self,
        release: &crate::utils::version_check::LatestPiRelease,
    ) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        let border = DynamicBorder::new(Rc::new(|text: &str| theme().fg("warning", text)));
        self.chat_container.add_child(Rc::new(border));
        let action = theme().fg("accent", &format!("{} update", crate::config::APP_NAME));
        let instruction = theme().fg(
            "muted",
            &format!("New version {} is available. Run ", release.version),
        ) + &action;
        let changelog = theme().fg("accent", "https://pi.dev/changelog");
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            format!(
                "{}\n{instruction}",
                bold(&theme().fg("warning", "Update Available"))
            ),
            1,
            0,
        ));
        self.chat_container.add_child(line);
        if let Some(note) = release
            .note
            .as_deref()
            .filter(|note| !note.trim().is_empty())
        {
            self.chat_container.add_child(Rc::new(Spacer::new()));
            self.chat_container.add_child(Rc::new(Markdown::new(
                note.trim(),
                1,
                0,
                self.markdown_theme_with_settings(),
            )));
            self.chat_container.add_child(Rc::new(Spacer::new()));
        }
        let changelog_line: Rc<dyn Component> = Rc::new(Text::with_padding(
            theme().fg("muted", "Changelog: ") + &changelog,
            1,
            0,
        ));
        self.chat_container.add_child(changelog_line);
        self.tui.request_render(false);
    }

    /// Show a package-update notification.
    pub(crate) fn show_package_update_notification(&self, packages: &[String]) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        let packages = packages
            .iter()
            .map(|package| format!("- {package}"))
            .collect::<Vec<_>>()
            .join("\n");
        let action = theme().fg(
            "accent",
            &format!("{} update --extensions", crate::config::APP_NAME),
        );
        let line: Rc<dyn Component> = Rc::new(Text::with_padding(
            format!(
                "{}\n{}\n{}\n{packages}",
                bold(&theme().fg("warning", "Package Updates Available")),
                theme().fg("muted", "Package updates are available. Run ") + &action,
                theme().fg("muted", "Packages:")
            ),
            1,
            0,
        ));
        self.chat_container.add_child(line);
        self.tui.request_render(false);
    }

    /// Reset the extension UI surface, upstream's `resetExtensionUI`.
    pub(crate) fn reset_extension_ui(&self) {
        self.hide_extension_selector();
        self.hide_extension_input();
        self.hide_extension_editor();
        self.tui.hide_overlay();
        self.clear_extension_terminal_input_listeners();
        self.set_extension_footer(None);
        self.set_extension_header(None);
        self.clear_extension_widgets();
        self.footer_data_provider.clear_extension_statuses();
        self.footer.invalidate();
        self.autocomplete_provider_wrappers.borrow_mut().clear();
        self.set_custom_editor_component(None);
        self.setup_autocomplete_provider();
        self.default_editor.set_on_extension_shortcut(None);
        self.update_terminal_title();
        *self.working_message.borrow_mut() = None;
        self.working_visible.set(true);
        self.set_working_indicator(None);
        self.set_hidden_thinking_label(None);
    }
}

/// The cache-statistics pricing view over the model runtime.
struct RuntimePrices(crate::model_runtime::ModelRuntime);

impl crate::cache_stats::ModelPriceSource for RuntimePrices {
    fn cache_read_price(&self, provider: &str, model_id: &str) -> Option<f64> {
        self.0
            .get_model(provider, model_id)
            .map(|model| model.cost.rates.cache_read)
    }
}
