//! Startup, run loop, renderer switching, rebinding, and shutdown.
//!
//! Partial port: the frame mounts, starts, binds extensions, and runs the
//! command/event pump. The remaining startup branches are recorded in the
//! ticket report.

use std::rc::Rc;
use std::time::Duration;

use pi_tui::tui::{Component, TuiMode, TuiStopOptions};

use crate::core::agent_session::ExtensionBindings;
use crate::extensions::reasons::ExtensionMode;
use crate::settings_manager::SettingsStorageKind;

use super::FrameCommand;
use super::InteractiveMode;

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Initialize the mounted frame, upstream's `init`.
    pub(crate) async fn init(self: &Rc<Self>) {
        if self.is_initialized.get() {
            return;
        }
        self.render_widgets();
        let viewport = crate::modes::interactive::chat_viewport::create_chat_viewport(
            crate::modes::interactive::chat_viewport::ChatViewportOptions {
                document: self.document_container.clone(),
                pending_messages: self.pending_messages_container.clone(),
                status: self.status_container.clone(),
                editor: self.editor_container.clone(),
                footer: self.footer_container.clone(),
                widgets_above: Some(self.widget_container_above.clone()),
                widgets_below: Some(self.widget_container_below.clone()),
                scrollbar: Some(
                    self.settings()
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get_fullscreen_scrollbar(),
                ),
                scrollbar_track_style: Some(std::sync::Arc::new(|text: &str| {
                    crate::modes::interactive::theme::theme::theme().fg("scrollbarTrack", text)
                })),
                scrollbar_thumb_style: Some(std::sync::Arc::new(|text: &str| {
                    crate::modes::interactive::theme::theme::theme().fg("scrollbarThumb", text)
                })),
            },
        );
        *self.transcript_scroll_view.borrow_mut() = Some(viewport.transcript);
        *self.fullscreen_layout_root.borrow_mut() = Some(viewport.root);
        *self.changelog_markdown.borrow_mut() = self.changelog_for_display();
        let components: Vec<Rc<dyn Component>> = vec![
            self.document_container.clone(),
            self.pending_messages_container.clone(),
            self.status_container.clone(),
            self.widget_container_above.clone(),
            self.editor_container.clone(),
            self.widget_container_below.clone(),
            self.footer_container.clone(),
        ];
        self.mount_interactive_tui(&components);
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
        self.default_editor.set_on_submit(Some(Rc::new({
            let commands = self.commands();
            move |text: &str| {
                let _ = commands.send(FrameCommand::Submit(text.to_owned()));
            }
        })));
        let editor_child: Rc<dyn Component> = self.default_editor.clone();
        self.tui.set_focus(Some(editor_child));
        self.tui.start();
        self.is_initialized.set(true);
        // Accept text while startup completes, but only report progress.
        self.default_editor.set_on_submit(Some(Rc::new({
            let commands = self.commands();
            move |text: &str| {
                let _ = commands.send(FrameCommand::StartupSubmit(text.to_owned()));
            }
        })));
        self.register_signal_handlers();
        self.theme_controller.apply_from_settings();
        let status_weak = Rc::downgrade(self);
        let on_status = move |status: &crate::utils::tools_manager::ToolStatus| {
            if let Some(mode) = status_weak.upgrade() {
                mode.show_managed_tool_status(status);
            }
        };
        let client = pi_ai::http::default_http_client();
        let fd = crate::utils::tools_manager::ensure_tool(
            &client,
            crate::utils::tools_manager::ManagedTool::Fd,
            Some(&on_status),
        )
        .await;
        let _ = crate::utils::tools_manager::ensure_tool(
            &client,
            crate::utils::tools_manager::ManagedTool::Rg,
            Some(&on_status),
        )
        .await;
        *self.fd_path.borrow_mut() = fd.map(std::path::PathBuf::from);
        self.setup_key_handlers();
        self.setup_editor_submit_handler();
        self.bind_current_session_extensions().await;
        self.show_loaded_resources(true);
        self.show_startup_notices_if_needed();
        self.render_initial_messages();
        self.update_terminal_title();
        self.tui.render_now(false);
    }

    /// Run the interactive loop. Returns the process exit code.
    pub async fn run(self: &Rc<Self>) -> i32 {
        self.drive(self.init()).await;
        self.announce_startup_state();
        self.drive(self.process_initial_messages()).await;
        loop {
            self.tui.poll(Duration::from_millis(8));
            self.drain_queues_and_commands().await;
            if self
                .shutdown_flag
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                self.shutdown().await;
            }
            if self.is_shutting_down.get() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        0
    }

    /// Await one frame future while keeping the TUI and extension dialogs
    /// alive.
    pub(crate) async fn drive<F>(&self, future: F) -> F::Output
    where
        F: std::future::Future,
    {
        tokio::pin!(future);
        loop {
            tokio::select! {
                output = &mut future => return output,
                () = tokio::time::sleep(Duration::from_millis(4)) => {}
            }
            self.tui.poll(Duration::ZERO);
            self.drain_events_and_ui();
        }
    }

    fn drain_events_and_ui(&self) {
        loop {
            let event = self
                .event_rx
                .borrow_mut()
                .as_mut()
                .and_then(|receiver| receiver.try_recv().ok());
            match event {
                Some(event) => self.handle_event(event),
                None => break,
            }
        }
        loop {
            let request = self
                .ui_requests
                .borrow_mut()
                .as_mut()
                .and_then(|receiver| receiver.try_recv().ok());
            match request {
                Some(request) => self.present_ui_request(request),
                None => break,
            }
        }
    }

    async fn drain_queues_and_commands(&self) {
        self.drain_events_and_ui();
        loop {
            let command = self
                .command_rx
                .borrow_mut()
                .as_mut()
                .and_then(|receiver| receiver.try_recv().ok());
            match command {
                Some(command) => self.handle_command(command).await,
                None => break,
            }
        }
    }

    pub(crate) async fn bind_current_session_extensions(&self) {
        let bindings = ExtensionBindings {
            mode: Some(ExtensionMode::Tui),
            ui_context: Some(
                self.ui_context.clone() as std::sync::Arc<dyn std::any::Any + Send + Sync>
            ),
            ..ExtensionBindings::default()
        };
        let session = self.session();
        if let Err(error) = self.drive(session.bind_extensions(bindings)).await {
            self.show_error(&format!("Failed to bind extensions: {error}"));
        }
    }

    /// Rebind the frame to the current session.
    pub(crate) async fn rebind_current_session(&self, _render_before_bind: bool) {
        self.bind_current_session_extensions().await;
        self.update_terminal_title();
    }

    /// Mount the frame components and the fullscreen layout root.
    /// Emit the startup diagnostics, warnings, and model fallback.
    fn announce_startup_state(&self) {
        let options = self.options.borrow().clone();
        for diagnostic in &options.startup_diagnostics {
            match diagnostic.kind {
                crate::core::agent_session_services::AgentSessionRuntimeDiagnosticKind::Error => {
                    self.show_error(&diagnostic.message);
                }
                crate::core::agent_session_services::AgentSessionRuntimeDiagnosticKind::Warning => {
                    self.show_warning(&diagnostic.message);
                }
                crate::core::agent_session_services::AgentSessionRuntimeDiagnosticKind::Info => {
                    self.show_status(&diagnostic.message);
                }
            }
        }
        if !options.migrated_providers.is_empty() {
            self.show_warning(&format!(
                "Migrated credentials to auth.json: {}",
                options.migrated_providers.join(", ")
            ));
        }
        if let Some(error) = self.session().model_runtime().get_error() {
            self.show_error(&format!("models.json error: {error}"));
        }
        if let Some(message) = &options.model_fallback_message {
            self.show_warning(message);
        }
    }

    async fn process_initial_messages(&self) {
        let options = self.options.borrow().clone();
        if let Some(message) = options.initial_message {
            let prompt = crate::core::agent_session::PromptOptions {
                images: Some(options.initial_images),
                ..crate::core::agent_session::PromptOptions::default()
            };
            if let Err(error) = self.session().prompt(message, Some(prompt)).await {
                self.show_error(&error);
            }
        }
        for message in options.initial_messages {
            if let Err(error) = self.session().prompt(message, None).await {
                self.show_error(&error);
            }
        }
    }

    pub(crate) fn mount_interactive_tui(&self, components: &[Rc<dyn Component>]) {
        for component in components {
            self.tui.add_child(Rc::clone(component));
        }
        let root = self.fullscreen_layout_root.borrow().clone();
        self.tui.set_layout_root(root);
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Switch the renderer between regular and fullscreen, upstream's
    /// `switchTuiMode`. The stable base keeps the mounted components; only the
    /// renderer strategy and the fullscreen layout root change.
    pub(crate) fn switch_tui_mode(
        &self,
        mode: TuiMode,
        restore_progress: bool,
        start_renderer: bool,
    ) -> bool {
        if mode == self.tui.mode() {
            return true;
        }
        if self.tui.has_overlay_entries() {
            return false;
        }
        let focus = self.tui.get_focused_component();
        if let Some(main) = self.main_screen.borrow().as_ref() {
            *self.main_screen_render_state.borrow_mut() = Some(main.capture_render_state());
        }
        self.tui.stop(TuiStopOptions {
            preserve_screen: true,
        });
        self.tui.set_focus(None);
        self.tui.clear();
        self.tui.set_layout_root(None);
        match mode {
            TuiMode::Fullscreen => {
                let alt = pi_tui::tui_alt_screen::TuiAltScreen::new(
                    pi_tui::tui_alt_screen::TuiAltScreenConfig {
                        copy_on_select: Some(
                            self.settings()
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .get_fullscreen_copy_on_select(),
                        ),
                        ..pi_tui::tui_alt_screen::TuiAltScreenConfig::default()
                    },
                );
                self.tui.set_renderer(Box::new(alt.clone()));
                *self.alt_screen.borrow_mut() = Some(alt);
                *self.main_screen.borrow_mut() = None;
            }
            TuiMode::Regular => {
                let main = Rc::new(pi_tui::tui_main_screen::TuiMainScreen::new(
                    pi_tui::tui_main_screen::TuiMainScreenConfig::default(),
                ));
                self.tui
                    .set_renderer(Box::new(MainRenderer(Rc::clone(&main))));
                *self.main_screen.borrow_mut() = Some(main);
                *self.alt_screen.borrow_mut() = None;
            }
        }
        self.options.borrow_mut().tui_mode = Some(mode);
        if self.tui.is_viewport_tui() {
            let root = self.fullscreen_layout_root.borrow().clone();
            self.tui.set_layout_root(root);
        } else if let Some(state) = self.main_screen_render_state.borrow().as_ref()
            && let Some(main) = self.main_screen.borrow().as_ref()
        {
            main.restore_render_state(state);
        }
        self.tui.invalidate();
        self.tui.set_focus(focus);
        if !start_renderer {
            return true;
        }
        self.tui.start();
        let theme_ui: Rc<dyn crate::modes::interactive::theme::theme_controller::ThemeUi> = Rc::new(
            crate::modes::interactive::theme::theme_controller::TuiThemeUi::new(Rc::clone(
                &self.tui,
            )),
        );
        self.theme_controller.rebind_tui(theme_ui);
        self.rebind_extension_terminal_input_listeners();
        if restore_progress
            && self
                .settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_show_terminal_progress()
            && (self.session().is_streaming() || self.session().is_compacting())
        {
            self.set_terminal_progress(true);
        }
        true
    }

    /// Toggle the terminal progress indicator through the terminal write
    /// seam.
    pub(crate) fn set_terminal_progress(&self, active: bool) {
        let sequence = if active {
            "\x1b]9;4;3\x07"
        } else {
            "\x1b]9;4;0\x07"
        };
        self.tui.terminal_write(sequence);
    }

    /// Write the terminal title.
    pub(crate) fn update_terminal_title(&self) {
        let session = self.session();
        let manager = crate::core::agent_session::lock(&session.session_manager);
        let cwd_basename = crate::utils::paths::basename_posix(manager.cwd());
        let name = manager.get_session_name();
        drop(manager);
        let title = name.map_or_else(
            || format!("{} - {cwd_basename}", crate::config::APP_TITLE),
            |name| format!("{} - {name} - {cwd_basename}", crate::config::APP_TITLE),
        );
        self.tui.terminal_write(&format!("\x1b]0;{title}\x07"));
    }

    /// Apply the settings that change while the frame runs.
    pub(crate) fn apply_runtime_settings(&self) {
        let (padding, visible, hide_thinking, output_pad, cursor, clear, capability) = {
            let settings = self.settings();
            let settings = settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                usize::try_from(settings.get_editor_padding_x()).unwrap_or_default(),
                usize::try_from(settings.get_autocomplete_max_visible()).unwrap_or(5),
                settings.get_hide_thinking_block(),
                usize::from(settings.get_output_pad()),
                settings.get_show_hardware_cursor(),
                settings.get_clear_on_shrink(),
                settings.get_terminal_capability_overrides(),
            )
        };
        pi_tui::terminal_image::set_capability_overrides(capability);
        self.default_editor.set_padding_x(padding);
        self.default_editor.set_autocomplete_max_visible(visible);
        self.editor().set_padding_x(padding);
        self.editor().set_autocomplete_max_visible(visible);
        self.hide_thinking_block.set(hide_thinking);
        self.output_pad.set(output_pad);
        self.tui.set_show_hardware_cursor(cursor);
        self.tui.set_clear_on_shrink(clear);
        if !clear && self.active_status_indicator.borrow().is_none() {
            self.status_container.clear();
        }
        self.footer.set_session(Rc::new(super::SessionFooterSource {
            session: self.session(),
        }));
        self.footer
            .set_auto_compact_enabled(self.session().auto_compaction_enabled());
        let session = self.session();
        let cwd = crate::core::agent_session::lock(&session.session_manager)
            .cwd()
            .to_owned();
        self.footer_data_provider.set_cwd(&cwd);
        self.apply_fullscreen_scrollbar_setting();
    }

    /// The changelog entries to show at startup.
    pub(crate) fn changelog_for_display(&self) -> Option<String> {
        if !self.session().agent().state().messages.is_empty() {
            return None;
        }
        let last = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_last_changelog_version();
        let entries =
            crate::utils::changelog::parse_changelog(&crate::config::get_changelog_path());
        let Some(last) = last else {
            self.settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .set_last_changelog_version(&self.version);
            return None;
        };
        let new_entries = crate::utils::changelog::get_new_entries(&entries, &last);
        if new_entries.is_empty() {
            return None;
        }
        self.settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_last_changelog_version(&self.version);
        Some(
            new_entries
                .into_iter()
                .map(|entry| {
                    crate::utils::changelog::normalize_changelog_links(
                        &entry.content,
                        &format!("{}.{}.{}", entry.major, entry.minor, entry.patch),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }

    fn register_signal_handlers(&self) {
        self.unregister_signal_handlers();
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            for kind in [SignalKind::terminate(), SignalKind::hangup()] {
                let Ok(mut stream) = signal(kind) else {
                    continue;
                };
                let flag = std::sync::Arc::clone(&self.shutdown_flag);
                let handle = tokio::spawn(async move {
                    if stream.recv().await.is_some() {
                        crate::utils::shell::kill_tracked_detached_children();
                        flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                self.signal_cleanup_handlers
                    .borrow_mut()
                    .push(Box::new(move || handle.abort()));
            }
        }
    }

    fn unregister_signal_handlers(&self) {
        for cleanup in self.signal_cleanup_handlers.borrow_mut().drain(..) {
            cleanup();
        }
    }

    /// Stop the frame and release its subscriptions.
    pub fn stop(&self, _fullscreen_exit_output: crate::settings_manager::FullscreenExitOutput) {
        self.dispose_active_selector();
        self.clear_status_indicator(None);
        self.theme_controller.disable_auto_sync();
        self.clear_extension_terminal_input_listeners();
        self.footer.dispose();
        self.footer_data_provider.dispose();
        if let Some(unsubscribe) = self.unsubscribe.borrow_mut().take() {
            unsubscribe();
        }
        if self.is_initialized.get() {
            self.tui.stop(TuiStopOptions {
                preserve_screen: self.tui.mode() == TuiMode::Fullscreen,
            });
            self.is_initialized.set(false);
        }
    }

    /// Shut the frame down.
    pub(crate) async fn shutdown(&self) {
        if self.is_shutting_down.get() {
            return;
        }
        self.is_shutting_down.set(true);
        self.stop(
            self.settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_fullscreen_exit_output(),
        );
        if let Some(mut runtime) = self.runtime_host.borrow_mut().take() {
            runtime.dispose().await;
        }
    }
}

/// The main-screen renderer behind the stable `Tui`.
#[derive(Debug)]
struct MainRenderer(Rc<pi_tui::tui_main_screen::TuiMainScreen>);

impl pi_tui::tui::TuiRenderer for MainRenderer {
    fn mode(&self) -> TuiMode {
        pi_tui::tui::TuiRenderer::mode(self.0.as_ref())
    }
    fn do_render(&self, tui: &pi_tui::tui::Tui) {
        pi_tui::tui::TuiRenderer::do_render(self.0.as_ref(), tui);
    }
    fn reset_render_state(&self) {
        pi_tui::tui::TuiRenderer::reset_render_state(self.0.as_ref());
    }
    fn before_terminal_stop(&self, tui: &pi_tui::tui::Tui, options: &TuiStopOptions) {
        pi_tui::tui::TuiRenderer::before_terminal_stop(self.0.as_ref(), tui, options);
    }
}
