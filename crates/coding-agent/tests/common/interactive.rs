//! The interactive-mode acceptance harness: a headless terminal, an
//! in-memory runtime over a scripted faux provider, and the helpers the ten
//! ported suites share.
//!
//! # Restatements against upstream's vitest fixtures
//!
//! - Upstream's `VirtualTerminal` (an `@xterm/headless` emulator) becomes
//!   [`RecordingTerminal`]: it records the start/stop/write stream and
//!   delivers pre-decoded input chunks to the handler `Tui::start` installs.
//!   The suites that inspect a screen assert on the raw write stream or on a
//!   container's `render`, not on an emulated grid; the frame tests never
//!   need cursor-addressed state.
//! - The process-global clipboard is swapped through
//!   [`pi_coding_agent::utils::clipboard::set_native_clipboard_override`],
//!   the port's existing process-wide seam for the clipboard module upstream's
//!   suites mock. [`ClipboardGuard`] serializes those suites through a
//!   process-wide lock and restores the previous override on drop.
//! - The session is a real `AgentSessionRuntime` over
//!   [`InMemorySettingsStorage`] and an in-memory model registry; the faux
//!   provider answers through a scripted stream. Upstream's per-test
//!   `vi.fn()` session doubles become real session state, and assertions read
//!   the session's own accessors.
//! - `Tui` is constructed through `InteractiveMode::new_with_terminal`, the
//!   narrow constructor seam this harness added; upstream's `createInteractiveTui`
//!   already accepted a terminal.
//! - The resource loader is a real `DefaultResourceLoader` whose overrides
//!   (`skills_override`, `extensions` load seam, `agents_files_override`, ...)
//!   carry the fixtures; upstream's fake loader object becomes these override
//!   closures.

#![allow(
    dead_code,
    unreachable_pub,
    reason = "each suite recompiles the shared module and consumes only the helpers it drives"
)]
#![allow(
    clippy::missing_fields_in_debug,
    clippy::too_many_lines,
    clippy::type_complexity,
    reason = "the fixture's Debug and builder read best in one place, and every suite recompiles it"
)]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pi_ai::types::Model;
use pi_coding_agent::core::agent_session_runtime::{
    AgentSessionRuntime, CreateAgentSessionRuntimeOptions, CreateAgentSessionRuntimeResult,
    create_agent_session_runtime,
};
use pi_coding_agent::core::agent_session_services::{
    CreateAgentSessionFromServicesOptions, CreateAgentSessionServicesOptions,
    create_agent_session_from_services, create_agent_session_services,
};
use pi_coding_agent::core::sdk::CreateAgentSessionError;
use pi_coding_agent::modes::interactive::interactive_mode::InteractiveMode;
use pi_coding_agent::resource_loader::DefaultResourceLoaderOptions;
use pi_coding_agent::session_manager::SessionManager;
use pi_coding_agent::settings_manager::{
    InMemorySettingsStorage, Settings, SettingsManager, SettingsManagerCreateOptions,
};
use pi_tui::terminal::{InputHandler, ResizeHandler, Terminal};
use pi_tui::tui::{TuiMode, TuiStopOptions};
use tempfile::TempDir;

/// The terminal operations one test observes.
#[derive(Default)]
struct TerminalState {
    writes: String,
    start_count: usize,
    stop_count: usize,
    progress: Vec<bool>,
    input_handler: Option<Box<dyn FnMut(String)>>,
    resize_handler: Option<ResizeHandler>,
    /// Chunks the terminal delivers on its next `poll`, the startup dialogs'
    /// only way to answer before they return.
    poll_inputs: std::collections::VecDeque<String>,
    /// Chunks held until their deadline, for prompts that start after a
    /// blocking theme probe.
    delayed_poll_inputs: std::collections::VecDeque<(std::time::Instant, String)>,
}

/// The headless recording terminal, upstream's `RecordingTerminal` over
/// `VirtualTerminal`.
#[derive(Clone)]
pub struct RecordingTerminal {
    state: Rc<RefCell<TerminalState>>,
    columns: Cell<u16>,
    rows: Cell<u16>,
}

impl std::fmt::Debug for RecordingTerminal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecordingTerminal")
            .field("writes", &self.state.borrow().writes.len())
            .field("starts", &self.state.borrow().start_count)
            .field("stops", &self.state.borrow().stop_count)
            .finish()
    }
}

impl RecordingTerminal {
    /// A 80x24 terminal, upstream's `VirtualTerminal` default.
    #[must_use]
    pub fn new(columns: u16, rows: u16) -> Self {
        Self {
            state: Rc::new(RefCell::new(TerminalState::default())),
            columns: Cell::new(columns),
            rows: Cell::new(rows),
        }
    }

    /// Deliver one decoded input chunk to the handler `Tui::start` installed,
    /// upstream's `VirtualTerminal.sendInput`.
    pub fn send_input(&self, data: &str) {
        let mut state = self.state.borrow_mut();
        if let Some(handler) = state.input_handler.as_mut() {
            handler(data.to_owned());
        }
    }

    /// Everything written so far.
    #[must_use]
    pub fn writes(&self) -> String {
        self.state.borrow().writes.clone()
    }

    /// The number of terminal starts.
    #[must_use]
    pub fn start_count(&self) -> usize {
        self.state.borrow().start_count
    }

    /// The number of terminal stops.
    #[must_use]
    pub fn stop_count(&self) -> usize {
        self.state.borrow().stop_count
    }

    /// The progress flags the terminal received, upstream's
    /// `setProgress` spy.
    #[must_use]
    pub fn progress(&self) -> Vec<bool> {
        self.state.borrow().progress.clone()
    }

    /// Deliver one chunk on the next `poll` instead of immediately. A caller
    /// that has to answer a dialog before its blocking prompt loop returns
    /// (the startup selectors) queues the input here.
    pub fn queue_poll_input(&self, data: &str) {
        self.state
            .borrow_mut()
            .poll_inputs
            .push_back(data.to_owned());
    }

    /// Deliver one chunk on the first `poll` at least `delay` from now.
    pub fn queue_poll_input_after(&self, delay: Duration, data: &str) {
        self.state
            .borrow_mut()
            .delayed_poll_inputs
            .push_back((std::time::Instant::now() + delay, data.to_owned()));
    }
}

impl Terminal for RecordingTerminal {
    fn start(&mut self, on_input: InputHandler, on_resize: ResizeHandler) {
        let mut state = self.state.borrow_mut();
        state.start_count += 1;
        state.input_handler = Some(Box::new(on_input));
        state.resize_handler = Some(on_resize);
    }

    fn stop(&mut self) {
        self.state.borrow_mut().stop_count += 1;
    }

    fn drain_input(&mut self, _max_ms: u64, _idle_ms: u64) {}

    fn write(&mut self, data: &str) {
        self.state.borrow_mut().writes.push_str(data);
    }

    fn columns(&self) -> u16 {
        self.columns.get()
    }

    fn rows(&self) -> u16 {
        self.rows.get()
    }

    fn is_kitty_protocol_active(&self) -> bool {
        false
    }

    fn move_by(&mut self, _lines: i32) {}

    fn hide_cursor(&mut self) {}

    fn show_cursor(&mut self) {}

    fn clear_line(&mut self) {}

    fn clear_from_cursor(&mut self) {}

    fn clear_screen(&mut self) {}

    fn set_title(&mut self, _title: &str) {}

    fn set_progress(&mut self, active: bool) {
        self.state.borrow_mut().progress.push(active);
    }

    fn poll(&mut self, _timeout: Duration) {
        let mut state = self.state.borrow_mut();
        let now = std::time::Instant::now();
        while let Some((due, _)) = state.delayed_poll_inputs.front() {
            if *due > now {
                break;
            }
            let (_, data) = state
                .delayed_poll_inputs
                .pop_front()
                .expect("the due chunk is present");
            state.poll_inputs.push_back(data);
        }
        while let Some(data) = state.poll_inputs.pop_front() {
            if let Some(handler) = state.input_handler.as_mut() {
                handler(data);
            }
        }
    }
}

/// Serialize the clipboard suites: the native-clipboard override is
/// process-global, and the acceptance suites run in parallel threads.
static CLIPBOARD_LOCK: Mutex<()> = Mutex::new(());

/// The clipboard double upstream's suites supply through `vi.mock`: it
/// records writes and returns the configured text.
pub struct RecordingClipboard {
    /// The text a read returns; `None` is an empty clipboard.
    pub text: Mutex<Option<String>>,
    /// Every text written.
    pub copied: Mutex<Vec<String>>,
    /// Every read.
    pub reads: Mutex<usize>,
    /// The image bytes an image read returns; `None` is empty.
    pub image: Mutex<Option<Vec<u8>>>,
}

impl pi_coding_agent::utils::clipboard::NativeClipboard for RecordingClipboard {
    fn get_text(&self) -> Result<String, pi_coding_agent::utils::clipboard::ClipboardError> {
        *self.reads.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        Ok(self
            .text
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .unwrap_or_default())
    }

    fn get_image(&self) -> Result<Vec<u8>, pi_coding_agent::utils::clipboard::ClipboardError> {
        Ok(self
            .image
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .unwrap_or_default())
    }

    fn set_text(&self, text: &str) -> bool {
        self.copied
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(text.to_owned());
        true
    }
}

/// Records the clipboard-helper invocations the Linux copy path makes.
struct RecordingCommandRunner {
    clipboard: Arc<RecordingClipboard>,
}

impl pi_coding_agent::utils::clipboard_command::ClipboardCommandRunner for RecordingCommandRunner {
    fn run<'a>(
        &'a self,
        _command: &'a str,
        _args: &'a [String],
        options: &'a pi_coding_agent::utils::clipboard_command::ClipboardCommandOptions,
    ) -> std::pin::Pin<Box<dyn Future<Output = Option<Vec<u8>>> + 'a>> {
        if let Some(input) = &options.input {
            self.clipboard
                .copied
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(input.clone());
        }
        Box::pin(async { Some(Vec::new()) })
    }
}

/// The clipboard override's lifetime: holds the process-wide lock and
/// restores the previous override on drop.
pub struct ClipboardGuard {
    clipboard: Arc<RecordingClipboard>,
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl ClipboardGuard {
    /// Install a clipboard seeded with `text`.
    #[must_use]
    pub fn new(text: Option<&str>) -> Self {
        Self::with_image(text, None)
    }

    /// Install a clipboard whose image read returns `image`.
    #[must_use]
    pub fn with_image(text: Option<&str>, image: Option<Vec<u8>>) -> Self {
        let serial = CLIPBOARD_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let clipboard = Arc::new(RecordingClipboard {
            text: Mutex::new(text.map(str::to_owned)),
            copied: Mutex::new(Vec::new()),
            reads: Mutex::new(0),
            image: Mutex::new(image),
        });
        let clipboard_for_override: Arc<dyn pi_coding_agent::utils::clipboard::NativeClipboard> =
            clipboard.clone();
        pi_coding_agent::utils::clipboard::set_native_clipboard_override(Some(
            clipboard_for_override,
        ));
        pi_coding_agent::utils::clipboard::set_clipboard_command_runner_override(Some(Arc::new(
            RecordingCommandRunner {
                clipboard: Arc::clone(&clipboard),
            },
        )));
        Self {
            clipboard,
            _serial: serial,
        }
    }

    /// The recorded writes.
    #[must_use]
    pub fn copied(&self) -> Vec<String> {
        self.clipboard
            .copied
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Clear the recorded writes, upstream's `copyToClipboard.mockClear()`.
    pub fn clear(&self) {
        self.clipboard
            .copied
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// The number of reads.
    #[must_use]
    pub fn reads(&self) -> usize {
        *self
            .clipboard
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for ClipboardGuard {
    fn drop(&mut self) {
        pi_coding_agent::utils::clipboard::set_native_clipboard_override(None);
        pi_coding_agent::utils::clipboard::set_clipboard_command_runner_override(None);
    }
}

/// The resource fixtures one frame's loader serves, upstream's
/// `createShowLoadedResourcesThis` option object.
#[derive(Clone, Default)]
pub struct ResourceFixtures {
    /// Skills, as `(name, file_path, source_info)`.
    pub skills: Vec<(String, String, pi_coding_agent::source_info::SourceInfo)>,
    /// Prompt templates, as `(name, file_path, source_info)`.
    pub prompts: Vec<(String, String, pi_coding_agent::source_info::SourceInfo)>,
    /// Extensions, as `(path, source_info)`.
    pub extensions: Vec<(String, Option<pi_coding_agent::source_info::SourceInfo>)>,
    /// Theme handles.
    pub themes: Vec<pi_coding_agent::resource_loader::ThemeHandle>,
    /// Context files.
    pub context_files: Vec<pi_coding_agent::resource_loader::ContextFile>,
    /// Skill diagnostics.
    pub skill_diagnostics: Vec<pi_coding_agent::diagnostics::ResourceDiagnostic>,
    /// Inline extensions, as `(name, hidden, factory)`.
    pub extension_factories: Vec<(
        Option<String>,
        bool,
        pi_coding_agent::extensions::api::ExtensionFactory,
    )>,
}

/// The resource-loader options the fixture's loader is built from.
fn loader_options(
    cwd: &str,
    agent_dir: &str,
    fixtures: Option<ResourceFixtures>,
) -> DefaultResourceLoaderOptions {
    let fixtures = fixtures.unwrap_or_default();
    let skills = fixtures.skills.clone();
    let prompts = fixtures.prompts.clone();
    let themes = fixtures.themes.clone();
    let context = fixtures.context_files.clone();
    let diagnostics = fixtures.skill_diagnostics.clone();
    let extensions = fixtures.extensions;
    let extension_factories = fixtures
        .extension_factories
        .iter()
        .map(
            |(name, hidden, factory)| pi_coding_agent::extensions::api::InlineExtension {
                name: name.clone(),
                hidden: *hidden,
                factory: Arc::clone(factory),
            },
        )
        .collect();
    DefaultResourceLoaderOptions {
        cwd: cwd.to_owned(),
        agent_dir: agent_dir.to_owned(),
        extension_factories,
        no_extensions: false,
        no_skills: false,
        no_prompt_templates: false,
        no_themes: false,
        no_context_files: false,
        load_extensions_fn: Some(Arc::new(move |_paths, _cwd, runtime| {
            let extensions = extensions.clone();
            Box::pin(async move {
                pi_coding_agent::resource_loader::ExtensionLoadResult {
                    extensions: extensions
                        .into_iter()
                        .map(|(path, source_info)| {
                            pi_coding_agent::resource_loader::ExtensionEntry {
                                resolved_path: path.clone(),
                                path,
                                hidden: false,
                                source_info,
                                record: None,
                            }
                        })
                        .collect(),
                    errors: Vec::new(),
                    runtime: runtime.unwrap_or_default(),
                }
            })
        })),
        skills_override: Some(Box::new(move |mut result| {
            result.skills = skills
                .iter()
                .map(
                    |(name, file_path, source_info)| pi_coding_agent::skills::Skill {
                        name: name.clone(),
                        description: format!("{name} skill"),
                        file_path: file_path.clone(),
                        base_dir: file_path
                            .rsplit_once('/')
                            .map_or_else(String::new, |(dir, _)| dir.to_owned()),
                        source_info: source_info.clone(),
                        disable_model_invocation: false,
                    },
                )
                .collect();
            result.diagnostics.extend(diagnostics.clone());
            result
        })),
        prompts_override: Some(Box::new(move |mut result| {
            result.prompts = prompts
                .iter()
                .map(|(name, file_path, source_info)| {
                    pi_coding_agent::prompt_templates::PromptTemplate {
                        name: name.clone(),
                        description: format!("{name} template"),
                        argument_hint: None,
                        content: String::new(),
                        source_info: source_info.clone(),
                        file_path: file_path.clone(),
                    }
                })
                .collect();
            result
        })),
        themes_override: Some(Box::new(move |mut result| {
            result.themes.clone_from(&themes);
            result
        })),
        agents_files_override: Some(Box::new(move |_files| context.clone())),
        ..DefaultResourceLoaderOptions::default()
    }
}

/// The session-manager seeder, a test's typed-entry fixture.
pub type ManagerSeeder = Box<dyn FnMut(&mut SessionManager)>;

/// The harness options, upstream's per-suite fixture configuration.
pub struct HarnessOptions {
    /// The initial settings.
    pub settings: Settings,
    /// The session model.
    pub model: Model,
    /// Additional models the runtime offers.
    pub extra_models: Vec<Model>,
    /// The seeded auth store.
    pub auth: pi_coding_agent::auth_storage::AuthStorageData,
    /// Messages the session manager starts with.
    pub messages: Vec<pi_agent_core::types::AgentMessage>,
    /// Resource fixtures the loader serves.
    pub fixtures: Option<ResourceFixtures>,
    /// The presentation mode.
    pub tui_mode: TuiMode,
    /// The terminal dimensions.
    pub terminal_size: (u16, u16),
    /// Mount over a persisted session store instead of an in-memory manager.
    pub persisted: bool,
    /// The faux provider's stream; `None` settles immediately.
    pub stream: Option<pi_agent_core::types::StreamFn>,
    /// Seed the session manager before the frame mounts; receives the fresh
    /// manager so a test can append typed entries with known ids.
    pub seed_manager: Option<ManagerSeeder>,
    /// The frame's verbose startup flag, upstream's `InteractiveModeOptions.verbose`.
    pub verbose: bool,
    /// Additional frame options (startup diagnostics, startup prompts, the
    /// deferred trust target). The harness overrides `verbose` and `tui_mode`
    /// from the dedicated fields above.
    pub frame_options:
        pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions,
    /// Mount over this project cwd instead of a fresh temp path.
    pub cwd_override: Option<String>,
    /// Leave the frame's `tui_mode` unset so it reads the settings' mode.
    pub use_settings_tui_mode: bool,
}

impl Default for HarnessOptions {
    fn default() -> Self {
        Self {
            settings: Settings::new(),
            model: super::sdk_fixture::faux_model(),
            extra_models: Vec::new(),
            auth: pi_coding_agent::auth_storage::AuthStorageData::new(),
            messages: Vec::new(),
            fixtures: None,
            tui_mode: TuiMode::Regular,
            terminal_size: (80, 24),
            persisted: false,
            stream: None,
            seed_manager: None,
            verbose: false,
            frame_options: pi_coding_agent::modes::interactive::interactive_mode::InteractiveModeOptions::default(),
            cwd_override: None,
            use_settings_tui_mode: false,
        }
    }
}

/// One mounted frame and the handles its test drives.
pub struct Harness {
    /// The frame under test.
    pub frame: Rc<InteractiveMode<InMemorySettingsStorage>>,
    /// The recording terminal the frame renders through.
    pub terminal: RecordingTerminal,
    /// The temp root.
    pub temp: TempDir,
    /// The project cwd.
    pub cwd: String,
    /// The agent dir.
    pub agent_dir: String,
    /// The settings manager the session sees.
    pub settings: Arc<Mutex<SettingsManager<InMemorySettingsStorage>>>,
}

impl Harness {
    /// Mount a frame with the given options.
    ///
    /// # Panics
    /// The fixture's own setup failures, which a test cannot satisfy.
    pub async fn new(mut options: HarnessOptions) -> Self {
        pi_coding_agent::modes::interactive::theme::theme::init_theme(Some("dark"), false);
        let temp = TempDir::new().expect("tempdir");
        let cwd = options
            .cwd_override
            .clone()
            .unwrap_or_else(|| temp.path().join("project").display().to_string());
        let agent_dir = temp.path().join("agent").display().to_string();
        std::fs::create_dir_all(&cwd).expect("cwd");
        std::fs::create_dir_all(&agent_dir).expect("agent dir");

        let registry = super::model_layer::create_in_memory_model_registry(
            super::model_layer::in_memory_auth_storage(&options.auth),
        )
        .await;
        let runtime = registry.runtime().clone();
        register_models(&runtime, &options);
        let settings_manager =
            SettingsManager::in_memory(&options.settings, SettingsManagerCreateOptions::default());
        let settings = Arc::new(Mutex::new(settings_manager));

        let mut manager = if options.persisted {
            SessionManager::create(&cwd, Some(&format!("{cwd}/sessions")), None).expect("session")
        } else {
            SessionManager::in_memory(Some(&cwd), None, None).expect("session")
        };
        if let Some(seed) = options.seed_manager.as_mut() {
            seed(&mut manager);
        }
        for message in &options.messages {
            manager.append_message(message.clone()).expect("message");
        }

        let settings_seed = options.settings.clone();
        let fixtures = options.fixtures.clone();
        let model = options.model.clone();
        let runtime_for_factory = runtime.clone();
        let cwd_for_factory = cwd.clone();
        let agent_dir_for_factory = agent_dir.clone();
        let factory: pi_coding_agent::core::agent_session_runtime::CreateAgentSessionRuntimeFactory<
            InMemorySettingsStorage,
        > = Arc::new(move |runtime_options| {
            let runtime = runtime_for_factory.clone();
            let fixtures = fixtures.clone();
            let model = model.clone();
            let settings_seed = settings_seed.clone();
            let loader = loader_options(
                &runtime_options.cwd,
                &runtime_options.agent_dir,
                fixtures,
            );
            Box::pin(async move {
                let services = create_agent_session_services::<InMemorySettingsStorage>(
                    CreateAgentSessionServicesOptions {
                        cwd: runtime_options.cwd.clone(),
                        agent_dir: Some(runtime_options.agent_dir.clone()),
                        settings_manager: Some(SettingsManager::in_memory(
                            &settings_seed,
                            SettingsManagerCreateOptions::default(),
                        )),
                        model_runtime: Some(runtime),
                        model_runtime_signal: None,
                        extension_flag_values: None,
                        resource_loader_options: Some(loader),
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
                        session_manager: runtime_options.session_manager,
                        session_start_event: runtime_options.session_start_event,
                        model: Some(model),
                        thinking_level: None,
                        scoped_models: None,
                        tools: Some(Vec::new()),
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
        });
        let _ = cwd_for_factory;
        let _ = agent_dir_for_factory;
        let runtime = create_agent_session_runtime(
            factory,
            CreateAgentSessionRuntimeOptions {
                cwd: cwd.clone(),
                agent_dir: agent_dir.clone(),
                session_manager: manager,
                session_start_event: None,
                project_trust_context: None,
            },
        )
        .await
        .expect("runtime");
        let terminal = RecordingTerminal::new(options.terminal_size.0, options.terminal_size.1);
        let mut frame_options = options.frame_options;
        if !options.use_settings_tui_mode {
            frame_options.tui_mode = Some(options.tui_mode);
        }
        frame_options.verbose = options.verbose;
        let frame = InteractiveMode::new_with_terminal(
            runtime,
            frame_options,
            Some(Box::new(terminal.clone())),
        );
        Self {
            frame,
            terminal,
            temp,
            cwd,
            agent_dir,
            settings,
        }
    }

    /// The rendered transcript text with ANSI stripped, upstream's
    /// `renderAll(chatContainer)` plus `stripAnsi`.
    #[must_use]
    pub fn chat_text(&self) -> String {
        render_container(&self.frame.chat_container(), 220)
    }

    /// The rendered loaded-resources text with ANSI stripped.
    #[must_use]
    pub fn resources_text(&self) -> String {
        render_container(&self.frame.loaded_resources_container(), 220)
    }

    /// Start the mounted TUI so key input reaches the focused component.
    pub fn start_tui(&self) {
        self.frame.tui().start();
    }

    /// Deliver one decoded input chunk to the focused component. The TUI must
    /// be started or the terminal has no handler yet.
    pub fn send_input(&self, data: &str) {
        self.terminal.send_input(data);
    }

    /// Deliver one chunk on the terminal's next `poll`, for dialogs whose
    /// prompt loop blocks before the test regains control.
    pub fn queue_poll_input(&self, data: &str) {
        self.terminal.queue_poll_input(data);
    }

    /// Poll the TUI once so queued terminal input lands and the scheduler
    /// repaints.
    pub fn pump_tui(&self) {
        self.frame.tui().poll(Duration::ZERO);
    }

    /// Stop the mounted TUI.
    pub fn stop_tui(&self) {
        self.frame.tui().stop(TuiStopOptions {
            preserve_screen: false,
        });
    }
}

/// Render every child of `container` and normalize the text the way the
/// suites assert on it: ANSI stripped, backslashes normalized, trailing
/// whitespace trimmed per line, whole string trimmed.
#[must_use]
pub fn render_container(container: &pi_tui::tui::Container, width: usize) -> String {
    container
        .children
        .borrow()
        .iter()
        .flat_map(|child| child.render(width))
        .map(|line| pi_coding_agent::utils::ansi::strip_ansi(&line))
        .collect::<Vec<_>>()
        .join("\n")
        .replace('\\', "/")
        .split('\n')
        .map(|line| line.trim_end().to_owned())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

/// Register the faux provider and every additional fixture provider.
fn register_models(
    runtime: &pi_coding_agent::model_runtime::ModelRuntime,
    options: &HarnessOptions,
) {
    use pi_ai::types::{Api, AssistantMessageEvent, StopReason};
    let settle = Arc::new(|model: &Model| {
        let stream = pi_ai::utils::event_stream::assistant_message_event_stream();
        let mut message = super::sdk_fixture::done_message();
        message.provider = model.provider.clone();
        message.model.clone_from(&model.id);
        message.api = model.api.clone();
        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message,
        });
        stream
    });
    let settle_for_faux = Arc::clone(&settle);
    let scripted_stream = options.stream.clone();
    runtime
        .register_provider(
            "faux",
            pi_coding_agent::provider_composer::ProviderConfigInput {
                base_url: Some("https://faux.invalid/v1".to_owned()),
                api: Some(Api::from("openai-completions")),
                api_key: Some("faux-key".to_owned()),
                stream_simple: Some(Arc::new(move |model, context, options| {
                    scripted_stream.as_ref().map_or_else(
                        || settle_for_faux(model),
                        |stream| stream(model, context, options),
                    )
                })),
                models: Some(vec![
                    provider_model_input(&super::sdk_fixture::faux_model()),
                    super::sdk_fixture::provider_model(
                        super::sdk_fixture::FAUX_SECOND_MODEL_ID,
                        true,
                        vec![pi_ai::types::Modality::Text],
                    ),
                ]),
                ..pi_coding_agent::provider_composer::ProviderConfigInput::default()
            },
        )
        .expect("faux provider registers");
    let mut by_provider: std::collections::BTreeMap<String, Vec<Model>> =
        std::collections::BTreeMap::new();
    for model in &options.extra_models {
        if model.provider.0 != "faux" {
            by_provider
                .entry(model.provider.0.clone())
                .or_default()
                .push(model.clone());
        }
    }
    // The anthropic provider comes from the builtin catalog with its OAuth
    // and api-key auth config; registering a fixture provider would replace
    // the very auth surface the warning suite checks.
    if options.model.provider.0 != "faux"
        && options.model.provider.0 != "anthropic"
        && !by_provider.contains_key(&options.model.provider.0)
    {
        by_provider
            .entry(options.model.provider.0.clone())
            .or_default()
            .push(options.model.clone());
    }
    for (provider, models) in by_provider {
        let settle_for_provider = Arc::clone(&settle);
        let api = models.first().map(|model| model.api.clone());
        let base_url = models.first().map(|model| model.base_url.clone());
        runtime
            .register_provider(
                &provider,
                pi_coding_agent::provider_composer::ProviderConfigInput {
                    name: Some(provider.clone()),
                    base_url,
                    api,
                    api_key: Some("test-key".to_owned()),
                    stream_simple: Some(Arc::new(move |model, _context, _options| {
                        settle_for_provider(model)
                    })),
                    models: Some(models.iter().map(provider_model_input).collect()),
                    ..pi_coding_agent::provider_composer::ProviderConfigInput::default()
                },
            )
            .expect("fixture provider registers");
    }
}

/// Project one model onto the provider registration input.
fn provider_model_input(model: &Model) -> pi_coding_agent::provider_composer::ProviderModelInput {
    pi_coding_agent::provider_composer::ProviderModelInput {
        id: model.id.clone(),
        name: model.name.clone(),
        api: Some(model.api.clone()),
        base_url: None,
        reasoning: model.reasoning,
        thinking_level_map: model.thinking_level_map.clone(),
        input: model.input.clone(),
        cost: model.cost.clone(),
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        sampling_params: model.sampling_params.clone(),
        headers: model.headers.clone(),
        compat: model.compat.clone(),
    }
}

/// A minimal app editor for the custom-editor seams: renders one label, keeps
/// its text, and records every embedded-indicator write.
pub struct TestEditor {
    label: String,
    text: RefCell<String>,
    history: RefCell<Vec<String>>,
    embed_working_status: bool,
    indicators: RefCell<Vec<bool>>,
}

impl TestEditor {
    /// A 25x80-style editor with the given label and embed opt-in.
    #[must_use]
    pub fn new(label: &str, embed_working_status: bool) -> Self {
        Self {
            label: label.to_owned(),
            text: RefCell::new(String::new()),
            history: RefCell::new(Vec::new()),
            embed_working_status,
            indicators: RefCell::new(Vec::new()),
        }
    }

    /// The recorded embedded-indicator writes, `true` for set, `false` for
    /// clear.
    #[must_use]
    pub fn indicator_writes(&self) -> Vec<bool> {
        self.indicators.borrow().clone()
    }
}

impl pi_tui::tui::Component for TestEditor {
    fn render(&self, _width: usize) -> Vec<String> {
        vec![self.label.clone()]
    }

    fn handle_input(&self, _data: &str) {}

    fn invalidate(&self) {}
}

impl pi_tui::components::EditorComponent for TestEditor {
    fn get_text(&self) -> String {
        self.text.borrow().clone()
    }

    fn set_text(&self, text: &str) {
        text.clone_into(&mut self.text.borrow_mut());
    }

    fn add_to_history(&self, text: &str) {
        self.history.borrow_mut().push(text.to_owned());
    }

    fn insert_text_at_cursor(&self, text: &str) {
        self.text.borrow_mut().push_str(text);
    }

    fn get_expanded_text(&self) -> Option<String> {
        None
    }

    fn set_padding_x(&self, _padding: usize) {}

    fn set_autocomplete_provider(
        &self,
        _provider: Arc<dyn pi_tui::autocomplete::AutocompleteProvider>,
    ) {
    }

    fn set_autocomplete_max_visible(&self, _max_visible: usize) {}
}

impl pi_coding_agent::modes::interactive::interactive_mode::AppEditor for TestEditor {
    fn set_working_status_indicator(
        &self,
        indicator: Option<
            Rc<
                dyn pi_coding_agent::modes::interactive::components::custom_editor::EmbeddedStatusIndicator,
            >,
        >,
    ) {
        self.indicators.borrow_mut().push(indicator.is_some());
    }

    fn embed_working_status(&self) -> bool {
        self.embed_working_status
    }
}
