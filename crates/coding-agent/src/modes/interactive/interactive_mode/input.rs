//! Key handlers, the editor submit dispatch, clipboard paste, bash, and
//! external editing.

use std::rc::Rc;

use crate::core::agent_session::PromptOptions;
use crate::extensions::reasons::StreamingBehavior;
use crate::settings_manager::SettingsStorageKind;

use super::{FrameCommand, InteractiveMode, QueueDelivery};

/// The current wall clock in milliseconds, upstream's `Date.now()`.
fn now_ms() -> i64 {
    i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
    )
    .unwrap_or(i64::MAX)
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Install the app key handlers, upstream's `setupKeyHandlers`.
    pub(crate) fn setup_key_handlers(&self) {
        self.default_editor.set_on_escape(Some(Rc::new({
            let commands = self.commands();
            move || {
                let _ = commands.send(FrameCommand::Escape);
            }
        })));
        self.default_editor.on_action(
            "app.clear",
            Rc::new({
                let commands = self.commands();
                move || {
                    let _ = commands.send(FrameCommand::CtrlC);
                }
            }),
        );
        self.default_editor.set_on_ctrl_d(Some(Rc::new({
            let commands = self.commands();
            move || {
                let _ = commands.send(FrameCommand::CtrlD);
            }
        })));
        let actions: [(&str, FrameCommand); 14] = [
            ("app.suspend", FrameCommand::Suspend),
            ("app.thinking.cycle", FrameCommand::CycleThinkingLevel),
            (
                "app.model.cycleForward",
                FrameCommand::CycleModel(crate::core::agent_session::CycleDirection::Forward),
            ),
            (
                "app.model.cycleBackward",
                FrameCommand::CycleModel(crate::core::agent_session::CycleDirection::Backward),
            ),
            ("app.model.select", FrameCommand::ShowModelSelector(None)),
            ("app.tools.expand", FrameCommand::ToggleToolOutputExpansion),
            (
                "app.thinking.toggle",
                FrameCommand::ToggleThinkingBlockVisibility,
            ),
            ("app.editor.external", FrameCommand::OpenExternalEditor),
            (
                "app.message.copy",
                FrameCommand::CopyMessage {
                    flash_confirmation: true,
                    prefer_selection: true,
                },
            ),
            ("app.message.followUp", FrameCommand::FollowUp),
            ("app.message.dequeue", FrameCommand::Dequeue),
            ("app.session.new", FrameCommand::HandleClear),
            ("app.session.tree", FrameCommand::ShowTreeSelector(None)),
            ("app.session.fork", FrameCommand::ShowUserMessageSelector),
        ];
        for (action, command) in actions {
            let commands = self.commands();
            self.default_editor.on_action(
                action,
                Rc::new(move || {
                    let _ = commands.send(command.clone());
                }),
            );
        }
        let commands = self.commands();
        self.default_editor.on_action(
            "app.session.resume",
            Rc::new(move || {
                let _ = commands.send(FrameCommand::ShowSessionSelector);
            }),
        );
        let commands = self.commands();
        self.default_editor
            .set_on_change(Some(Rc::new(move |text: &str| {
                let _ = commands.send(FrameCommand::EditorChanged(text.to_owned()));
            })));
        let commands = self.commands();
        self.default_editor
            .set_on_paste_image(Some(Rc::new(move || {
                let _ = commands.send(FrameCommand::ClipboardPaste);
            })));
    }

    /// Install the submit handler, upstream's `setupEditorSubmitHandler`.
    pub(crate) fn setup_editor_submit_handler(&self) {
        let commands = self.commands();
        self.default_editor
            .set_on_submit(Some(Rc::new(move |text: &str| {
                let _ = commands.send(FrameCommand::Submit(text.to_owned()));
            })));
    }

    /// Handle Ctrl+C: clear the editor, or exit on a second press.
    pub(crate) async fn handle_ctrl_c(&self) {
        let now = now_ms();
        if now - self.last_sigint_time.get() < 500 {
            self.shutdown().await;
        } else {
            self.clear_editor();
            self.last_sigint_time.set(now);
        }
    }

    /// Handle Ctrl+D.
    pub(crate) async fn handle_ctrl_d(&self) {
        self.shutdown().await;
    }

    /// Handle Ctrl+Z.
    ///
    /// Restatement: the port has no process-group signal seam, so suspension
    /// reports the platform status instead of raising `SIGTSTP`.
    pub(crate) fn handle_ctrl_z(&self) {
        self.show_status("Suspend to background is not supported in this build");
    }

    /// Handle Escape, upstream's assigned `onEscape`.
    pub(crate) async fn handle_escape(&self) {
        if self.session().is_streaming() {
            self.restore_queued_messages_to_editor(true, None);
        } else if self.session().is_bash_running() {
            self.session().abort_bash();
        } else if self.is_bash_mode.get() {
            self.editor().set_text("");
            self.is_bash_mode.set(false);
            self.update_editor_border_color();
        } else if self.editor().get_text().trim().is_empty() {
            let action = self
                .settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_double_escape_action()
                .to_owned();
            if action != "none" {
                let now = now_ms();
                if now - self.last_escape_time.get() < 500 {
                    if action == "tree" {
                        let _ = self.commands().send(FrameCommand::ShowTreeSelector(None));
                    } else {
                        let _ = self.commands().send(FrameCommand::ShowUserMessageSelector);
                    }
                    self.last_escape_time.set(0);
                } else {
                    self.last_escape_time.set(now);
                }
            }
        }
    }

    /// Update the bash-mode border when the editor text changes.
    pub(crate) fn handle_editor_changed(&self, text: &str) {
        let was_bash = self.is_bash_mode.get();
        self.is_bash_mode.set(text.trim_start().starts_with('!'));
        if was_bash != self.is_bash_mode.get() {
            self.update_editor_border_color();
        }
    }

    /// Paste the clipboard's text into the focused component.
    pub(crate) async fn handle_right_click_paste(&self) {
        let Some(target) = self.tui.get_focused_component() else {
            return;
        };
        if let Some(text) = crate::utils::clipboard::read_clipboard_text().await {
            let data = format!("\x1b[200~{text}\x1b[201~");
            target.handle_input(&data);
            self.tui.request_render(false);
        }
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Paste the clipboard's image or text into the editor.
    pub(crate) async fn handle_clipboard_paste(&self) {
        let env = crate::config::default_env_lookup();
        let platform = crate::utils::clipboard::Platform::of_process();
        if let Ok(Some(image)) =
            crate::utils::clipboard_image::read_clipboard_image(&env, platform).await
        {
            let extension =
                crate::utils::clipboard_image::extension_for_image_mime_type(&image.mime_type)
                    .unwrap_or("png");
            let path = std::env::temp_dir()
                .join(format!("pi-clipboard-{}.{extension}", uuid::Uuid::new_v4()));
            if std::fs::write(&path, &image.bytes).is_ok() {
                self.editor()
                    .insert_text_at_cursor(&path.display().to_string());
                self.tui.request_render(false);
                return;
            }
        }
        if let Some(text) = crate::utils::clipboard::read_clipboard_text().await {
            self.editor().insert_text_at_cursor(&text);
            self.tui.request_render(false);
        }
    }

    /// Open the external editor over the current prompt.
    pub(crate) async fn handle_open_external_editor(&self) {
        let command = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_external_editor_command();
        let content = self
            .editor()
            .get_expanded_text()
            .unwrap_or_else(|| self.editor().get_text());
        self.tui.stop(pi_tui::tui::TuiStopOptions {
            preserve_screen: true,
        });
        let result = crate::modes::interactive::external_editor::edit_in_external_editor(
            crate::modes::interactive::external_editor::ExternalEditorOptions { command, content },
        )
        .await;
        self.tui.start();
        if let Ok(crate::modes::interactive::external_editor::ExternalEditorResult::Complete {
            content,
        }) = result
        {
            self.editor().set_text(&content);
        }
        self.tui.request_render(true);
    }

    /// Queue the editor's text as a follow-up, upstream's `handleFollowUp`.
    pub(crate) async fn handle_follow_up(&self) {
        let text = self
            .editor()
            .get_expanded_text()
            .unwrap_or_else(|| self.editor().get_text())
            .trim()
            .to_owned();
        if text.is_empty() {
            return;
        }
        if self.session().is_compacting() {
            if self.is_extension_command(&text) {
                self.editor().add_to_history(&text);
                self.editor().set_text("");
                let _ = self.session().prompt(text, None).await;
            } else {
                self.queue_compaction_message(&text, QueueDelivery::FollowUp);
            }
            return;
        }
        if self.session().is_streaming() {
            self.editor().add_to_history(&text);
            self.editor().set_text("");
            let options = PromptOptions {
                streaming_behavior: Some(StreamingBehavior::FollowUp),
                ..PromptOptions::default()
            };
            let _ = self.session().prompt(text, Some(options)).await;
            self.update_pending_messages_display();
            self.tui.request_render(false);
        } else {
            self.editor().set_text("");
            self.submit_text(&text).await;
        }
    }

    /// Restore every queued message to the editor.
    pub(crate) fn handle_dequeue(&self) {
        let restored = self.restore_queued_messages_to_editor(false, None);
        if restored == 0 {
            self.show_status("No queued messages to restore");
        } else {
            self.show_status(&format!(
                "Restored {restored} queued message{} to editor",
                if restored > 1 { "s" } else { "" }
            ));
        }
    }

    /// Whether the text names a registered extension command.
    pub(crate) fn is_extension_command(&self, text: &str) -> bool {
        if !text.starts_with('/') {
            return false;
        }
        let name = text
            .trim_start_matches('/')
            .split(' ')
            .next()
            .unwrap_or_default();
        self.session()
            .extension_runner()
            .get_command(name)
            .is_some()
    }

    /// Copy the last assistant message, upstream's `handleCopyCommand`.
    pub(crate) async fn handle_copy_command(&self, flash: bool, prefer_selection: bool) {
        if prefer_selection
            && let Some(alt) = self.alt_screen.borrow().as_ref()
            && !alt.get_copy_on_select()
            && alt.has_active_selection()
        {
            let _ = alt.copy_active_selection_to_clipboard();
            return;
        }
        let Some(text) = self.session().last_assistant_text() else {
            self.show_error("No agent messages to copy yet.");
            return;
        };
        match crate::utils::clipboard::copy_to_clipboard(&text).await {
            Ok(()) => {
                if flash && let Some(alt) = self.alt_screen.borrow().as_ref() {
                    alt.flash("Copied!", None);
                } else {
                    self.show_status("Copied last agent message to clipboard");
                }
            }
            Err(error) => self.show_error(&error.to_string()),
        }
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Dispatch one submitted editor text, upstream's `onSubmit` routing.
    pub(crate) async fn submit_text(&self, text: &str) {
        let text = text.trim().to_owned();
        if text.is_empty() {
            return;
        }
        if let Some(command) = self.route_slash_command(&text) {
            let _ = self.commands().send(command);
            return;
        }
        if text.starts_with('!') {
            let excluded = text.starts_with("!!");
            let command = if excluded {
                text.trim_start_matches("!!").trim()
            } else {
                text.trim_start_matches('!').trim()
            };
            if !command.is_empty() {
                if self.session().is_bash_running() {
                    self.show_warning(
                        "A bash command is already running. Press Esc to cancel it first.",
                    );
                    self.editor().set_text(&text);
                    return;
                }
                self.editor().add_to_history(&text);
                self.handle_bash_command(command, excluded).await;
                self.is_bash_mode.set(false);
                self.update_editor_border_color();
                return;
            }
        }
        if self.session().is_compacting() {
            if self.is_extension_command(&text) {
                self.editor().add_to_history(&text);
                self.editor().set_text("");
                let _ = self.session().prompt(text, None).await;
            } else {
                self.queue_compaction_message(&text, QueueDelivery::Steer);
            }
            return;
        }
        if self.session().is_streaming() {
            self.editor().add_to_history(&text);
            self.editor().set_text("");
            let options = PromptOptions {
                streaming_behavior: Some(StreamingBehavior::Steer),
                ..PromptOptions::default()
            };
            let session = self.session();
            if let Err(error) = self.drive(session.prompt(text, Some(options))).await {
                self.show_error(&error);
            }
            self.update_pending_messages_display();
            self.tui.request_render(false);
            return;
        }
        self.flush_pending_bash_components();
        self.editor().add_to_history(&text);
        self.editor().set_text("");
        let session = self.session();
        if let Err(error) = self.drive(session.prompt(text, None)).await {
            self.show_error(&error);
        }
        self.tui.request_render(false);
    }

    /// Map one slash line to its frame command, upstream's inline routing.
    fn route_slash_command(&self, text: &str) -> Option<FrameCommand> {
        let (command, argument) = text
            .split_once(' ')
            .map_or((text, None), |(command, argument)| {
                (command, Some(argument.trim().to_owned()))
            });
        let command = match command {
            "/settings" => FrameCommand::ShowSettingsSelector,
            "/scoped-models" => FrameCommand::ShowModelsSelector,
            "/model" => FrameCommand::HandleModelCommand(argument),
            "/thinking" => FrameCommand::HandleThinkingCommand(argument),
            "/export" => FrameCommand::HandleExport(text.to_owned()),
            "/import" => FrameCommand::HandleImport(text.to_owned()),
            "/share" => FrameCommand::HandleShare,
            "/copy" => FrameCommand::CopyMessage {
                flash_confirmation: false,
                prefer_selection: false,
            },
            "/name" => FrameCommand::HandleName(text.to_owned()),
            "/session" => FrameCommand::HandleSessionInfo,
            "/changelog" => FrameCommand::HandleChangelog,
            "/hotkeys" => FrameCommand::HandleHotkeys,
            "/fork" => FrameCommand::ShowUserMessageSelector,
            "/clone" => FrameCommand::HandleClone,
            "/tree" => FrameCommand::ShowTreeSelector(None),
            "/trust" => FrameCommand::ShowTrustSelector,
            "/login" => FrameCommand::HandleLogin(argument),
            "/logout" => FrameCommand::ShowOAuthSelector(false),
            "/new" => FrameCommand::HandleClear,
            "/compact" => FrameCommand::HandleCompact(argument),
            "/reload" => FrameCommand::HandleReload,
            "/debug" => FrameCommand::HandleDebug,
            "/arminsayshi" => FrameCommand::ArminSaysHi,
            "/dementedelves" => FrameCommand::DementedDelves,
            "/resume" => FrameCommand::ShowSessionSelector,
            "/quit" => FrameCommand::Quit,
            _ => return None,
        };
        Some(command)
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Dispatch one frame command.
    pub(crate) async fn handle_command(&self, command: FrameCommand) {
        match command {
            FrameCommand::Submit(text) => self.submit_text(&text).await,
            FrameCommand::StartupSubmit(text) => self.handle_startup_submit(&text),
            FrameCommand::Escape => self.handle_escape().await,
            FrameCommand::EditorChanged(text) => self.handle_editor_changed(&text),
            FrameCommand::CtrlC => self.handle_ctrl_c().await,
            FrameCommand::CtrlD => self.handle_ctrl_d().await,
            FrameCommand::Suspend => self.handle_ctrl_z(),
            FrameCommand::FollowUp => self.handle_follow_up().await,
            FrameCommand::Dequeue => self.handle_dequeue(),
            FrameCommand::ClipboardPaste => self.handle_clipboard_paste().await,
            FrameCommand::RightClickPaste => self.handle_right_click_paste().await,
            FrameCommand::ExtensionShortcutError(message) => self.show_error(&message),
            FrameCommand::ExtensionError { path, error, stack } => {
                self.show_extension_error(&path, &error, stack.as_deref());
            }
            FrameCommand::ToggleToolOutputExpansion => self.toggle_tool_output_expansion(),
            FrameCommand::ToggleThinkingBlockVisibility => self.toggle_thinking_block_visibility(),
            FrameCommand::OpenExternalEditor => self.handle_open_external_editor().await,
            FrameCommand::CopyMessage {
                flash_confirmation,
                prefer_selection,
            } => {
                self.handle_copy_command(flash_confirmation, prefer_selection)
                    .await;
            }
            FrameCommand::CycleThinkingLevel => self.cycle_thinking_level(),
            FrameCommand::CycleModel(direction) => self.cycle_model(direction).await,
            FrameCommand::ShowSettingsSelector => self.show_settings_selector(),
            FrameCommand::ShowModelsSelector => self.show_models_selector().await,
            FrameCommand::ShowModelSelector(search) => self.show_model_selector(search.as_deref()),
            FrameCommand::ShowThinkingSelector => self.show_thinking_selector(),
            FrameCommand::ShowTreeSelector(selected) => {
                self.show_tree_selector(selected.as_deref()).await;
            }
            FrameCommand::ShowUserMessageSelector => self.show_user_message_selector(),
            FrameCommand::ShowSessionSelector => self.show_session_selector().await,
            FrameCommand::ShowTrustSelector => self.show_trust_selector(),
            FrameCommand::ShowOAuthSelector(login) => self.show_oauth_selector(login).await,
            FrameCommand::LogoutProvider(provider) => self.logout_provider(&provider).await,
            FrameCommand::LoginFinished {
                provider_name,
                auth_type,
                result,
            } => self.finish_login(&provider_name, auth_type, result).await,
            FrameCommand::HandleModelCommand(search) => self.handle_model_command(search).await,
            FrameCommand::HandleThinkingCommand(search) => {
                self.handle_thinking_command(search.as_deref());
            }
            FrameCommand::SelectThinkingLevel { level, persist } => {
                self.select_thinking_level(level, persist);
            }
            FrameCommand::SetModel { model, persist } => self.set_model(model, persist).await,
            FrameCommand::HandleExport(text) => self.handle_export_command(&text).await,
            FrameCommand::HandleImport(text) => self.handle_import_command(&text).await,
            FrameCommand::HandleShare => self.handle_share_command().await,
            FrameCommand::HandleName(text) => self.handle_name_command(&text),
            FrameCommand::HandleSessionInfo => self.handle_session_command(),
            FrameCommand::HandleChangelog => self.handle_changelog_command(),
            FrameCommand::HandleHotkeys => self.handle_hotkeys_command(),
            FrameCommand::HandleClear => self.handle_clear_command().await,
            FrameCommand::HandleClone => self.handle_clone_command().await,
            FrameCommand::HandleReload => self.handle_reload_command().await,
            FrameCommand::HandleDebug => self.handle_debug_command(),
            FrameCommand::HandleLogin(provider) => {
                self.handle_login_command(provider.as_deref()).await;
            }
            FrameCommand::HandleCompact(instructions) => {
                self.handle_compact_command(instructions).await;
            }
            FrameCommand::HandleBash {
                command,
                exclude_from_context,
            } => {
                self.handle_bash_command(&command, exclude_from_context)
                    .await;
            }
            FrameCommand::HandleStartProviderLogin(provider) => {
                self.start_provider_login(provider).await;
            }
            FrameCommand::SetScopedModels(enabled) => self.apply_scoped_models(enabled),
            FrameCommand::PersistScopedModels(enabled) => self.persist_scoped_models(enabled),
            FrameCommand::ForkEntry(entry) => self.fork_entry(&entry, false).await,
            FrameCommand::NavigateTree(entry) => self.navigate_tree(&entry).await,
            FrameCommand::SetTheme(name) => self.apply_theme_setting(&name),
            FrameCommand::PreviewTheme(name) => self.theme_controller.preview(&name),
            FrameCommand::SetTuiMode(mode) => self.apply_tui_mode(mode),
            FrameCommand::RebuildChat => self.rebuild_chat_from_messages(),
            FrameCommand::RefreshAutocomplete => self.setup_autocomplete_provider(),
            FrameCommand::NotifyTrustSaved { trusted } => self.show_status(&format!(
                "Saved trust decision: {}. Restart {} for this to take effect.",
                if trusted { "trusted" } else { "untrusted" },
                crate::config::APP_NAME
            )),
            FrameCommand::SwitchSession(path) => {
                let _ = self.switch_session(&path).await;
            }
            FrameCommand::Quit => self.shutdown().await,
            FrameCommand::ArminSaysHi => self.handle_armin_says_hi(),
            FrameCommand::DementedDelves => self.handle_demented_delves(),
        }
    }

    /// Run one bash command, upstream's `handleBashCommand`.
    pub(crate) async fn handle_bash_command(&self, command: &str, exclude_from_context: bool) {
        let session = self.session();
        let _ = session
            .extension_runner()
            .emit_user_bash(command.to_owned())
            .await;
        let is_deferred = session.is_streaming();
        let component = Rc::new(
            crate::modes::interactive::components::bash_execution::BashExecutionComponent::new(
                command,
                self.render_request(),
                exclude_from_context,
            ),
        );
        if is_deferred {
            self.pending_messages_container
                .add_child(Rc::clone(&component) as Rc<dyn pi_tui::tui::Component>);
            self.pending_bash_components
                .borrow_mut()
                .push(Rc::clone(&component));
        } else {
            self.chat_container
                .add_child(Rc::clone(&component) as Rc<dyn pi_tui::tui::Component>);
        }
        *self.bash_component.borrow_mut() = Some(Rc::clone(&component));
        self.tui.request_render(false);
        match session
            .execute_bash(
                command,
                None,
                Some(crate::core::agent_session::BashExecutionOptions {
                    exclude_from_context,
                    ..crate::core::agent_session::BashExecutionOptions::default()
                }),
            )
            .await
        {
            Ok(result) => {
                if !result.output.is_empty() {
                    component.append_output(&result.output);
                }
                component.set_complete(
                    result.exit_code,
                    result.cancelled,
                    None,
                    result.full_output_path.as_deref(),
                );
            }
            Err(error) => {
                component.set_complete(None, false, None, None);
                self.show_error(&format!("Bash command failed: {error}"));
            }
        }
        *self.bash_component.borrow_mut() = None;
        self.tui.request_render(false);
    }

    /// Run compaction, upstream's `handleCompactCommand`.
    pub(crate) async fn handle_compact_command(&self, instructions: Option<String>) {
        self.clear_status_indicator(None);
        let _ = self.session().compact(instructions).await;
    }
}
