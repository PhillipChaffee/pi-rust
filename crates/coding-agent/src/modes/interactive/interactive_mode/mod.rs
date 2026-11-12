//! The interactive app frame, upstream's
//! `src/modes/interactive/interactive-mode.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! [`InteractiveMode`] owns the mounted component tree, the agent
//! subscription, every input handler, and the `AgentSessionRuntime` the mode
//! drives. Business logic stays on the session.
//!
//! # Split
//!
//! - this file — the struct, construction, and the shared editor adapter;
//! - `init` — startup, the run loop, renderer switching, rebinding,
//!   shutdown, and signal handling;
//! - `events` — the session-event switch;
//! - `input` — key handlers, submit dispatch, clipboard, bash, external
//!   editing;
//! - `display` — transcript rendering, status lines, notifications,
//!   queued messages, extension widgets/header/footer;
//! - `extensions` — the extension UI context and its dialogs;
//! - `commands` — selectors, login, and slash-command handlers.
//!
//! # Restatements
//!
//! - The frame is thread-bound while `AgentSession` is shared: every piece a
//!   pi-tui callback captures sits behind `Rc`/`RefCell`, and the owned
//!   `AgentSessionRuntime` sits in a take/put slot.
//! - The session listener cannot borrow the frame, so it forwards events
//!   through an unbounded channel the run loop drains.
//! - Runtime replacements are followed by an explicit frame rebind instead of
//!   a `setRebindSession` callback: that callback must be `Send + Sync`, and
//!   the frame is `!Send`.
//! - Input handlers, selector callbacks, and extension dialogs enqueue a
//!   `FrameCommand`; the run loop performs work that needs the runtime host
//!   or an await.
//! - Upstream duck-types optional editor members; the port carries them on
//!   `AppEditor`, implemented by the default editor and extension
//!   factories.
//! - The status wrappers build their own `StatusIndicator` in the component
//!   slice. The frame needs one `Rc<StatusIndicator>` for both the
//!   `Component` and `EmbeddedStatusIndicator` contracts, so it builds the
//!   base indicator with the same kind, colors, and message; retry's live
//!   countdown timer is not armed here.
//! - Terminal title and progress write the same OSC sequences through
//!   `Tui::terminal_write`; upstream's `ui.terminal` accessor has no port
//!   counterpart. The pre-stop `drainInput(1000)` is omitted because `Tui`
//!   owns the terminal and its stop path already disables the Kitty protocol
//!   and drops buffered input.
//! - `EditorComponent` carries no getters for padding or autocomplete rows;
//!   the custom-editor swap copies those values from settings.
//! - The extension registration channels for shortcuts, Markdown
//!   transformers, and custom message/entry renderers live on
//!   [`crate::extensions::runtime::Extension`] and are read through
//!   `ExtensionRunner` getters. Upstream's collision diagnostics against the
//!   resolved builtin bindings are not ported: the frame matches keys itself,
//!   so a shortcut that shadows an app binding simply wins in the editor.
//! - Registered themes are re-loaded from each handle's `source_path` because
//!   `ThemeHandle` carries no constructed theme.
//! - Upstream's `getUserInput` promise becomes the editor submit channel:
//!   the run loop drains the same text the promise would have resolved.
//! - The startup resource listing renders the same compact and expanded
//!   bodies upstream builds ([`render_loaded_resource_sections`]); the
//!   diagnostic groups keep the port's flat path/message form instead of the
//!   per-scope tree.
//! - Upstream scans `chatContainer.children` for `setExpanded` on each tool
//!   toggle; the port records every expandable row it mounts in a weak list
//!   (`Expandable`) because the component trait has no downcast.
//! - The acceptance suites drive the private methods upstream's vitest tests
//!   reach through `Reflect.get`, so those methods and the narrow state
//!   accessors they read are `pub` here.
//! - `checkForPackageUpdates`, `checkTmuxKeyboardSetup`,
//!   `reportInstallTelemetry`, and the Node terminal-crash handlers
//!   (`emergencyTerminalExit`, `uncaughtCrash`) have no process seam here and
//!   are not wired.
//! - `AgentSession` exposes no transport setter, so the transport setting
//!   persists and applies on the next session.
//! - `switchTuiMode` keeps one `Rc<Tui>` and swaps the renderer through
//!   `Tui::set_renderer`; components holding the base stay valid.

// The frame is a 1:1 port of a 4,000-line class: the mechanical style lints
// (line counts, single-pattern matches, explicit coercions, redundant clones)
// fire on the upstream shape rather than on a defect. Correctness lints stay
// on; the listed families are the ones the pinned shape cannot avoid.
#![allow(
    trivial_casts,
    unused_qualifications,
    dead_code,
    clippy::too_many_lines,
    clippy::unused_async,
    clippy::unused_async_trait_impl,
    clippy::unused_self,
    clippy::unnecessary_filter_map,
    clippy::single_match_else,
    clippy::semicolon_if_nothing_returned,
    clippy::redundant_pub_crate,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_clone,
    clippy::range_plus_one,
    clippy::needless_pass_by_value,
    clippy::needless_borrow,
    clippy::missing_fields_in_debug,
    clippy::missing_const_for_fn,
    clippy::match_same_arms,
    clippy::map_unwrap_or,
    clippy::map_flatten,
    clippy::let_and_return,
    clippy::large_futures,
    clippy::large_enum_variant,
    clippy::iter_with_drain,
    clippy::format_push_string,
    clippy::expect_used,
    clippy::doc_markdown,
    clippy::collapsible_if,
    clippy::assigning_clones,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::await_holding_refcell_ref,
    reason = "the pinned upstream shape carries these style patterns; the frame keeps them 1:1"
)]

mod commands;
mod display;
mod events;
mod extensions;
mod init;
mod input;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, PoisonError};

use pi_agent_core::types::ThinkingLevel;
use pi_ai::types::{ImageContent, Model};
use pi_tui::autocomplete::AutocompleteProvider;
use pi_tui::components::editor::{EditorCallback, EditorColorFn, EditorOptions, EditorTheme};
use pi_tui::components::loader::LoaderIndicatorOptions;
use pi_tui::components::scroll_view::ScrollView;
use pi_tui::components::{Spacer, Text};
use pi_tui::tui::Container;
use pi_tui::tui::{Component, ListenerId, Tui, TuiInputListener, TuiMode};
use pi_tui::tui_alt_screen::TuiAltScreen;
use pi_tui::tui_main_screen::{TuiMainScreen, TuiMainScreenRenderState};
use tokio::sync::mpsc;

use crate::config::{VERSION, get_agent_dir};
use crate::core::agent_session::{AgentSession, AgentSessionEvent, CycleDirection, lock};
use crate::core::agent_session_runtime::AgentSessionRuntime;
use crate::core::agent_session_services::AgentSessionRuntimeDiagnostic;
use crate::core::keybindings::AppKeybindingsManager;
use crate::extensions::types::MarkdownTransformer;
use crate::footer_data_provider::FooterDataProvider;
use crate::modes::interactive::components::assistant_message::AssistantMessageComponent;
use crate::modes::interactive::components::bash_execution::BashExecutionComponent;
use crate::modes::interactive::components::custom_editor::{
    AppActionHandler, CustomEditor, CustomEditorOptions, EmbeddedStatusIndicator,
    ExtensionShortcutHandler,
};
use crate::modes::interactive::components::extension_editor::ExtensionEditorComponent;
use crate::modes::interactive::components::extension_input::ExtensionInputComponent;
use crate::modes::interactive::components::extension_selector::ExtensionSelectorComponent;
use crate::modes::interactive::components::footer::{
    FooterComponent, FooterData, FooterModel, FooterSession,
};
use crate::modes::interactive::components::mermaid::create_mermaid_markdown_transformer;
use crate::modes::interactive::components::status_indicator::{IdleStatus, StatusIndicator};
use crate::modes::interactive::components::tool_execution::ToolExecutionComponent;
use crate::modes::interactive::theme::theme::{get_editor_theme, set_registered_themes, theme};
use crate::modes::interactive::theme::theme_controller::{
    InteractiveThemeController, ThemeControllerOptions, ThemeUi, TuiThemeUi,
};
use crate::modes::interactive::tui_renderer::{InteractiveTuiOptions, create_interactive_tui};
use crate::session_manager::FileEntry;
use crate::settings_manager::{SettingsManager, SettingsStorageKind};

use self::display::Expandable;

pub use self::display::{
    LoadedPromptEntry, LoadedResource, LoadedResourceSection, LoadedResourcesOptions,
    LoadedResourcesView, LoadedSkillEntry, LoadedThemeEntry, render_loaded_resource_sections,
};
pub use self::extensions::InteractiveUiContext;

/// The shared settings slot, replaced when a session replacement swaps the
/// manager.
pub(crate) type SettingsSlot<S> = Arc<Mutex<Arc<Mutex<SettingsManager<S>>>>>;

/// Options for [`InteractiveMode`], upstream's `InteractiveModeOptions`.
#[derive(Clone, Debug, Default)]
pub struct InteractiveModeOptions {
    /// Providers migrated to `auth.json`; the frame warns about them.
    pub migrated_providers: Vec<String>,
    /// Diagnostics collected before the TUI existed.
    pub startup_diagnostics: Vec<AgentSessionRuntimeDiagnostic>,
    /// The warning shown when the session model could not be restored.
    pub model_fallback_message: Option<String>,
    /// A cwd to trust after reload when it gained project resources.
    pub auto_trust_on_reload_cwd: Option<String>,
    /// The first message to send on startup.
    pub initial_message: Option<String>,
    /// Images attached to the first message.
    pub initial_images: Vec<ImageContent>,
    /// Messages sent after the first one.
    pub initial_messages: Vec<String>,
    /// Force the verbose startup listing.
    pub verbose: bool,
    /// The TUI layout mode; absent reads the settings' mode.
    pub tui_mode: Option<TuiMode>,
    /// The theme setting supplied for this invocation.
    pub initial_theme_setting: Option<String>,
}

/// A message queued while compaction runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionQueuedMessage {
    /// The queued text.
    pub text: String,
    /// How the message delivers after compaction.
    pub mode: QueueDelivery,
}

/// The delivery mode of a compaction-queued message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueDelivery {
    /// Send the message while the agent run continues.
    Steer,
    /// Send the message after the current run.
    FollowUp,
}

/// One deferred frame action, the restatement of upstream's inline async
/// closures.
#[derive(Clone)]
pub(crate) enum FrameCommand {
    Submit(String),
    StartupSubmit(String),
    Escape,
    EditorChanged(String),
    ClipboardPaste,
    RightClickPaste,
    ExtensionShortcutError(String),
    ExtensionError {
        path: String,
        error: String,
        stack: Option<String>,
    },
    FollowUp,
    Dequeue,
    CtrlC,
    CtrlD,
    Suspend,
    ToggleToolOutputExpansion,
    ToggleThinkingBlockVisibility,
    OpenExternalEditor,
    CopyMessage {
        flash_confirmation: bool,
        prefer_selection: bool,
    },
    CycleThinkingLevel,
    CycleModel(CycleDirection),
    ShowSettingsSelector,
    ShowModelsSelector,
    ShowModelSelector(Option<String>),
    ShowThinkingSelector,
    ShowTreeSelector(Option<String>),
    ShowUserMessageSelector,
    ShowSessionSelector,
    ShowTrustSelector,
    ShowOAuthSelector(bool),
    LogoutProvider(String),
    LoginFinished {
        provider_name: String,
        auth_type: pi_ai::auth::types::AuthType,
        result: Result<(), String>,
    },
    HandleModelCommand(Option<String>),
    HandleThinkingCommand(Option<String>),
    SelectThinkingLevel {
        level: ThinkingLevel,
        persist: bool,
    },
    SetModel {
        model: Model,
        persist: bool,
    },
    HandleExport(String),
    HandleImport(String),
    HandleShare,
    HandleName(String),
    HandleSessionInfo,
    HandleChangelog,
    HandleHotkeys,
    HandleClear,
    HandleClone,
    HandleReload,
    HandleDebug,
    HandleLogin(Option<String>),
    HandleCompact(Option<String>),
    HandleBash {
        command: String,
        exclude_from_context: bool,
    },
    HandleStartProviderLogin(
        crate::modes::interactive::components::oauth_selector::AuthSelectorProvider,
    ),
    SetScopedModels(Option<Vec<String>>),
    PersistScopedModels(Option<Vec<String>>),
    ForkEntry(String),
    NavigateTree(String),
    NotifyTrustSaved {
        trusted: bool,
    },
    SetTheme(String),
    PreviewTheme(String),
    SetTuiMode(TuiMode),
    RebuildChat,
    RefreshAutocomplete,
    SwitchSession(String),
    Quit,
    ArminSaysHi,
    DementedDelves,
}

/// A UI task the extension context marshals onto the frame loop.
pub(crate) enum UiRequest {
    Select {
        title: String,
        options: Vec<String>,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
        reply: tokio::sync::oneshot::Sender<Option<String>>,
    },
    Confirm {
        title: String,
        message: String,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
        reply: tokio::sync::oneshot::Sender<bool>,
    },
    Input {
        title: String,
        placeholder: Option<String>,
        signal: Option<pi_agent_core::harness::context::AbortSignal>,
        timeout: Option<u64>,
        reply: tokio::sync::oneshot::Sender<Option<String>>,
    },
    Editor {
        title: String,
        prefill: Option<String>,
        reply: tokio::sync::oneshot::Sender<Option<String>>,
    },
    Notify {
        message: String,
        kind: Option<UiNotifyKind>,
    },
    SetStatus {
        key: String,
        text: Option<String>,
    },
    SetWorkingMessage(Option<String>),
    SetWorkingVisible(bool),
    SetWorkingIndicator(Option<LoaderIndicatorOptions>),
    SetHiddenThinkingLabel(Option<String>),
    SetWidget {
        key: String,
        lines: Option<Vec<String>>,
        placement: UiWidgetPlacement,
    },
    SetTitle(String),
    PasteToEditor(String),
    SetEditorText(String),
    AuthPrompt {
        prompt: pi_ai::auth::types::AuthPrompt,
        reply: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    AuthNotify(pi_ai::auth::types::AuthEvent),

    GetEditorText(tokio::sync::oneshot::Sender<String>),
    GetToolsExpanded(tokio::sync::oneshot::Sender<bool>),
    SetToolsExpanded(bool),
    SetTheme(String, tokio::sync::oneshot::Sender<ThemeUpdateResult>),
    GetAllThemes(tokio::sync::oneshot::Sender<Vec<(String, String)>>),
    GetTheme(
        String,
        tokio::sync::oneshot::Sender<Option<Arc<crate::modes::interactive::theme::theme::Theme>>>,
    ),
}

/// The notify severity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UiNotifyKind {
    Info,
    Warning,
    Error,
}

/// Where an extension widget mounts, upstream's `placement`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum UiWidgetPlacement {
    #[default]
    AboveEditor,
    BelowEditor,
}

/// The result of a theme write, upstream's `ThemeResult`.
#[derive(Clone, Debug, Default)]
pub struct ThemeUpdateResult {
    /// Whether the theme was applied.
    pub success: bool,
    /// The failure message when it was not.
    pub error: Option<String>,
}

/// An extension editor factory, upstream's `EditorFactory`.
pub type EditorFactory =
    Rc<dyn Fn(&Rc<Tui>, EditorTheme, Rc<AppKeybindingsManager>) -> Rc<dyn AppEditor>>;

/// An autocomplete-provider wrapper, upstream's
/// `AutocompleteProviderFactory`.
pub type AutocompleteProviderFactory =
    Rc<dyn Fn(Arc<dyn AutocompleteProvider>) -> Arc<dyn AutocompleteProvider>>;

/// The app-level editor surface, upstream's duck-typed `EditorComponent`
/// plus the app-only members the frame assigns.
pub trait AppEditor: pi_tui::components::EditorComponent {
    /// Replace the border color function, upstream's `borderColor`.
    fn set_border_color(&self, _color: EditorColorFn) {}
    /// The current border color function.
    fn border_color(&self) -> Option<EditorColorFn> {
        None
    }
    /// Install the indicator drawn inside the top border.
    fn set_working_status_indicator(&self, _indicator: Option<Rc<dyn EmbeddedStatusIndicator>>) {}
    /// Whether the editor renders a working indicator in its border.
    fn embed_working_status(&self) -> bool {
        false
    }
    /// Install the extension shortcut handler.
    fn set_on_extension_shortcut(&self, _handler: Option<ExtensionShortcutHandler>) {}
    /// Install the Escape handler.
    fn set_on_escape(&self, _handler: Option<AppActionHandler>) {}
    /// The installed Escape handler.
    fn on_escape(&self) -> Option<AppActionHandler> {
        None
    }
    /// Install the Ctrl+D handler.
    fn set_on_ctrl_d(&self, _handler: Option<AppActionHandler>) {}
    /// Install the clipboard-paste handler.
    fn set_on_paste_image(&self, _handler: Option<AppActionHandler>) {}
    /// Install the submit callback.
    fn set_on_submit(&self, _callback: Option<EditorCallback>) {}
    /// Install the change callback.
    fn set_on_change(&self, _callback: Option<EditorCallback>) {}
    /// Register one app action handler.
    fn on_action(&self, _action: &str, _handler: AppActionHandler) {}
}

/// The default editor as an `AppEditor`.
pub(crate) struct DefaultEditor {
    inner: Rc<CustomEditor>,
}

impl DefaultEditor {
    fn new(editor: Rc<CustomEditor>) -> Self {
        Self { inner: editor }
    }
}

impl std::fmt::Debug for DefaultEditor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DefaultEditor")
            .finish_non_exhaustive()
    }
}

impl Component for DefaultEditor {
    fn render(&self, width: usize) -> Vec<String> {
        self.inner.render(width)
    }
    fn handle_input(&self, data: &str) {
        self.inner.handle_input(data);
    }
    fn wants_input(&self) -> bool {
        self.inner.wants_input()
    }
    fn handle_mouse(
        &self,
        event: &pi_tui::tui::TuiMouseEvent,
    ) -> Option<pi_tui::tui::TuiMouseEventResult> {
        self.inner.handle_mouse(event)
    }
    fn invalidate(&self) {
        self.inner.invalidate();
    }
    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.inner.children()
    }
}

impl pi_tui::components::EditorComponent for DefaultEditor {
    fn get_text(&self) -> String {
        self.inner.get_text()
    }
    fn set_text(&self, text: &str) {
        self.inner.set_text(text);
    }
    fn add_to_history(&self, text: &str) {
        self.inner.add_to_history(text);
    }
    fn insert_text_at_cursor(&self, text: &str) {
        self.inner.insert_text_at_cursor(text);
    }
    fn get_expanded_text(&self) -> Option<String> {
        self.inner.get_expanded_text()
    }
    fn set_padding_x(&self, padding: usize) {
        self.inner.set_padding_x(padding);
    }
    fn set_autocomplete_provider(&self, provider: Arc<dyn AutocompleteProvider>) {
        self.inner.set_autocomplete_provider(provider);
    }
    fn set_autocomplete_max_visible(&self, max_visible: usize) {
        self.inner.set_autocomplete_max_visible(max_visible);
    }
}

impl AppEditor for DefaultEditor {
    fn set_border_color(&self, color: EditorColorFn) {
        self.inner.set_border_color(color);
    }
    fn border_color(&self) -> Option<EditorColorFn> {
        Some(self.inner.border_color())
    }
    fn set_working_status_indicator(&self, indicator: Option<Rc<dyn EmbeddedStatusIndicator>>) {
        self.inner.set_working_status_indicator(indicator);
    }
    fn embed_working_status(&self) -> bool {
        self.inner.embed_working_status
    }
    fn set_on_extension_shortcut(&self, handler: Option<ExtensionShortcutHandler>) {
        self.inner.set_on_extension_shortcut(handler);
    }
    fn set_on_escape(&self, handler: Option<AppActionHandler>) {
        self.inner.set_on_escape(handler);
    }
    fn on_escape(&self) -> Option<AppActionHandler> {
        self.inner.on_escape()
    }
    fn set_on_ctrl_d(&self, handler: Option<AppActionHandler>) {
        self.inner.set_on_ctrl_d(handler);
    }
    fn set_on_paste_image(&self, handler: Option<AppActionHandler>) {
        self.inner.set_on_paste_image(handler);
    }
    fn set_on_submit(&self, callback: Option<EditorCallback>) {
        self.inner.set_on_submit(callback);
    }
    fn set_on_change(&self, callback: Option<EditorCallback>) {
        self.inner.set_on_change(callback);
    }
    fn on_action(&self, action: &str, handler: AppActionHandler) {
        self.inner.on_action(action, handler);
    }
}

/// The footer's session read surface over the shared session handle.
struct SessionFooterSource<S: SettingsStorageKind> {
    session: Arc<AgentSession<S>>,
}

impl<S: SettingsStorageKind> FooterSession for SessionFooterSource<S> {
    fn model(&self) -> Option<FooterModel> {
        let model = self.session.model()?;
        Some(FooterModel {
            id: model.id,
            provider: model.provider.0,
            context_window: model.context_window,
            reasoning: model.reasoning,
        })
    }
    fn thinking_level(&self) -> ThinkingLevel {
        self.session.thinking_level()
    }
    fn context_usage(&self) -> Option<crate::extensions::runtime::ContextUsage> {
        self.session.context_usage()
    }
    fn cwd(&self) -> String {
        self.session.cwd().to_owned()
    }
    fn session_name(&self) -> Option<String> {
        self.session.session_name()
    }
    fn is_using_subscription(&self, provider: &str) -> bool {
        self.session.model_runtime().is_using_subscription(provider)
    }
    fn for_each_entry(&self, visit: &mut dyn FnMut(&FileEntry)) {
        let manager = lock(&self.session.session_manager);
        for entry in manager.entries() {
            visit(entry);
        }
    }
}

/// The interactive app frame.
pub struct InteractiveMode<S: SettingsStorageKind> {
    runtime_host: RefCell<Option<AgentSessionRuntime<S>>>,
    settings: SettingsSlot<S>,
    tui: Rc<Tui>,
    alt_screen: RefCell<Option<TuiAltScreen>>,
    main_screen: RefCell<Option<Rc<TuiMainScreen>>>,
    main_screen_render_state: RefCell<Option<TuiMainScreenRenderState>>,
    loaded_resources_container: Rc<Container>,
    chat_container: Rc<Container>,
    document_container: Rc<Container>,
    transcript_scroll_view: RefCell<Option<Rc<ScrollView>>>,
    fullscreen_layout_root: RefCell<Option<Rc<dyn Component>>>,
    pending_messages_container: Rc<Container>,
    status_container: Rc<Container>,
    default_editor: Rc<CustomEditor>,
    editor: RefCell<Rc<dyn AppEditor>>,
    editor_component_factory: RefCell<Option<EditorFactory>>,
    autocomplete_provider: RefCell<Option<Arc<dyn AutocompleteProvider>>>,
    autocomplete_provider_wrappers: RefCell<Vec<AutocompleteProviderFactory>>,
    fd_path: RefCell<Option<PathBuf>>,
    editor_container: Rc<Container>,
    active_selector_token: RefCell<Option<u64>>,
    active_selector_dispose: RefCell<Option<Rc<dyn Fn()>>>,
    footer: Rc<FooterComponent>,
    footer_container: Rc<Container>,
    footer_data_provider: Rc<FooterDataProvider>,
    keybindings: Rc<AppKeybindingsManager>,
    version: String,
    is_initialized: Cell<bool>,
    active_status_indicator: RefCell<Option<Rc<StatusIndicator>>>,
    active_working_indicator_embedded: Cell<bool>,
    idle_status: Rc<IdleStatus>,
    working_message: RefCell<Option<String>>,
    working_visible: Cell<bool>,
    working_indicator_options: RefCell<Option<LoaderIndicatorOptions>>,
    hidden_thinking_label: RefCell<String>,
    last_sigint_time: Cell<i64>,
    last_escape_time: Cell<i64>,
    changelog_markdown: RefCell<Option<String>>,
    startup_notices_shown: Cell<bool>,
    anthropic_subscription_warning_shown: Cell<bool>,
    last_status_spacer: RefCell<Option<Rc<Spacer>>>,
    last_status_text: RefCell<Option<Rc<Text>>>,
    /// The expandable transcript rows the frame created, upstream scans
    /// `chatContainer.children` for `setExpanded` each toggle.
    expandable_rows: RefCell<Vec<std::rc::Weak<dyn Expandable>>>,
    managed_tool_status_started: Cell<bool>,
    streaming_component: RefCell<Option<Rc<AssistantMessageComponent>>>,
    assistant_rows: RefCell<Vec<Rc<AssistantMessageComponent>>>,
    streaming_message: RefCell<Option<pi_ai::types::AssistantMessage>>,
    pending_tools: RefCell<HashMap<String, Rc<ToolExecutionComponent>>>,
    tool_output_expanded: Cell<bool>,
    hide_thinking_block: Cell<bool>,
    output_pad: Cell<usize>,
    mermaid_markdown_transformer: MarkdownTransformer,
    skill_commands: RefCell<HashMap<String, String>>,
    signal_cleanup_handlers: RefCell<Vec<Box<dyn Fn()>>>,
    is_bash_mode: Cell<bool>,
    bash_component: RefCell<Option<Rc<BashExecutionComponent>>>,
    pending_bash_components: RefCell<Vec<Rc<BashExecutionComponent>>>,
    auto_compaction_escape_handler: RefCell<Option<Rc<dyn Fn()>>>,
    retry_escape_handler: RefCell<Option<Rc<dyn Fn()>>>,
    compaction_queued_messages: RefCell<Vec<CompactionQueuedMessage>>,
    shutdown_requested: Cell<bool>,
    is_shutting_down: Cell<bool>,
    extension_selector: RefCell<Option<Rc<ExtensionSelectorComponent>>>,
    extension_input: RefCell<Option<Rc<ExtensionInputComponent>>>,
    extension_editor: RefCell<Option<Rc<ExtensionEditorComponent>>>,
    extension_terminal_input_subscriptions: RefCell<Vec<(ListenerId, TuiInputListener)>>,
    extension_widgets_above: RefCell<BTreeMap<String, Rc<dyn Component>>>,
    extension_widgets_below: RefCell<BTreeMap<String, Rc<dyn Component>>>,
    widget_container_above: Rc<Container>,
    widget_container_below: Rc<Container>,
    custom_footer: RefCell<Option<Rc<dyn Component>>>,
    header_container: Rc<Container>,
    built_in_header: RefCell<Option<Rc<dyn Component>>>,
    custom_header: RefCell<Option<Rc<dyn Component>>>,
    login_dialog: RefCell<
        Option<Rc<crate::modes::interactive::components::login_dialog::LoginDialogComponent>>,
    >,
    options: RefCell<InteractiveModeOptions>,
    auto_trust_on_reload_cwd: RefCell<Option<String>>,
    theme_controller: InteractiveThemeController<S>,
    unsubscribe: RefCell<Option<Box<dyn FnOnce()>>>,
    event_sender: tokio::sync::mpsc::UnboundedSender<AgentSessionEvent>,
    shutdown_flag: Arc<std::sync::atomic::AtomicBool>,
    command_tx: mpsc::UnboundedSender<FrameCommand>,
    command_rx: RefCell<Option<mpsc::UnboundedReceiver<FrameCommand>>>,
    event_rx: RefCell<Option<mpsc::UnboundedReceiver<AgentSessionEvent>>>,
    ui_requests: RefCell<Option<mpsc::UnboundedReceiver<UiRequest>>>,
    ui_context: Arc<extensions::InteractiveUiContext>,
}

impl<S: SettingsStorageKind> std::fmt::Debug for InteractiveMode<S> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InteractiveMode")
            .field("version", &self.version)
            .field("initialized", &self.is_initialized.get())
            .field("tui_mode", &self.tui.mode())
            .finish_non_exhaustive()
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Build the frame over `runtime_host`.
    ///
    /// # Panics
    ///
    /// Never: poisoned settings locks are recovered.
    #[must_use]
    pub fn new(runtime_host: AgentSessionRuntime<S>, options: InteractiveModeOptions) -> Rc<Self> {
        Self::new_with_terminal(runtime_host, options, None)
    }

    /// Build the frame over `runtime_host`, rendering through `terminal`.
    ///
    /// Upstream's `createInteractiveTui` takes the terminal as an option; the
    /// frame's constructor did not previously expose it. Tests supply a
    /// headless terminal here so key input and rendered writes stay off the
    /// process TTY.
    ///
    /// # Panics
    ///
    /// Never: poisoned settings locks are recovered.
    #[must_use]
    pub fn new_with_terminal(
        runtime_host: AgentSessionRuntime<S>,
        options: InteractiveModeOptions,
        terminal: Option<Box<dyn pi_tui::terminal::Terminal>>,
    ) -> Rc<Self> {
        let session = Arc::clone(runtime_host.session());
        let settings: SettingsSlot<S> = Arc::new(Mutex::new(session.settings_manager()));
        let initial_settings = session.settings_manager();
        // Read every setting before the TUI options: a guard held in one
        // struct-literal field stays alive until the statement ends, so a
        // second lock in the next field would deadlock.
        let (tui_mode, show_hardware_cursor, fullscreen_copy_on_select, clear_on_shrink) = {
            let settings = initial_settings
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            (
                options.tui_mode.unwrap_or_else(|| settings.get_tui_mode()),
                settings.get_show_hardware_cursor(),
                settings.get_fullscreen_copy_on_select(),
                settings.get_clear_on_shrink(),
            )
        };
        let capability_overrides = initial_settings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_terminal_capability_overrides();
        let mut options = options;
        options.tui_mode = Some(tui_mode);

        pi_tui::terminal_image::set_capability_overrides(capability_overrides);

        let interactive = create_interactive_tui(InteractiveTuiOptions {
            tui_mode,
            show_hardware_cursor,
            log_directory: get_agent_dir().display().to_string(),
            terminal,
            on_right_click_paste: None,
            fullscreen_copy_on_select: Some(fullscreen_copy_on_select),
        });
        let tui = interactive.tui();
        let alt_screen = interactive.viewport().cloned();
        let main_screen = interactive.main_screen().cloned();
        tui.set_clear_on_shrink(clear_on_shrink);

        let header_container = Rc::new(Container::default());
        let loaded_resources_container = Rc::new(Container::default());
        let chat_container = Rc::new(Container::default());
        let document_container = Rc::new(Container::default());
        let header_child: Rc<dyn Component> = header_container.clone();
        let resources_child: Rc<dyn Component> = loaded_resources_container.clone();
        let chat_child: Rc<dyn Component> = chat_container.clone();
        document_container.add_child(header_child);
        document_container.add_child(resources_child);
        document_container.add_child(chat_child);
        let pending_messages_container = Rc::new(Container::default());
        let status_container = Rc::new(Container::default());
        let widget_container_above = Rc::new(Container::default());
        let widget_container_below = Rc::new(Container::default());

        let keybindings = Rc::new(AppKeybindingsManager::create(&get_agent_dir()));
        pi_tui::keybindings::set_keybindings(
            pi_tui::keybindings::KeybindingsManager::with_user_bindings(
                crate::core::keybindings::keybindings(),
                keybindings.get_user_bindings(),
            ),
        );

        let (editor_padding_x, autocomplete_max_visible) = {
            let settings = initial_settings
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            (
                usize::try_from(settings.get_editor_padding_x()).unwrap_or_default(),
                usize::try_from(settings.get_autocomplete_max_visible()).unwrap_or(5),
            )
        };
        let default_editor = Rc::new(CustomEditor::with_options(
            &tui,
            get_editor_theme(),
            Rc::clone(&keybindings),
            CustomEditorOptions {
                editor: EditorOptions {
                    padding_x: editor_padding_x,
                    autocomplete_max_visible,
                },
                embed_working_status: true,
            },
        ));
        let editor: Rc<dyn AppEditor> = Rc::new(DefaultEditor::new(Rc::clone(&default_editor)));
        let editor_container = Rc::new(Container::default());
        let editor_child: Rc<dyn Component> = editor.clone();
        editor_container.add_child(editor_child);

        let cwd = lock(&session.session_manager).cwd().to_owned();
        let footer_data_provider = Rc::new(FooterDataProvider::new(&cwd));
        let footer_session: Rc<dyn FooterSession> = Rc::new(SessionFooterSource {
            session: Arc::clone(&session),
        });
        let footer_data: Rc<dyn FooterData> = footer_data_provider.clone();
        let footer = Rc::new(FooterComponent::new(footer_session, footer_data));
        footer.set_auto_compact_enabled(session.auto_compaction_enabled());
        let footer_container = Rc::new(Container::default());
        let footer_child: Rc<dyn Component> = footer.clone();
        footer_container.add_child(footer_child);

        set_registered_themes(load_registered_themes(&session)).ok();

        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (ui_tx, ui_requests) = mpsc::unbounded_channel();
        let ui_context = extensions::InteractiveUiContext::new(ui_tx);
        let event_sender = event_tx.clone();
        let unsubscribe = session.subscribe(Arc::new(move |event| {
            let _ = event_tx.send(event);
        }));

        let (hide_thinking_block, output_pad) = {
            let settings = initial_settings
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            (
                settings.get_hide_thinking_block(),
                usize::from(settings.get_output_pad()),
            )
        };
        let mermaid_settings = Arc::clone(&settings);
        let mermaid_markdown_transformer = create_mermaid_markdown_transformer(
            Arc::new(move || {
                let manager = mermaid_settings
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clone();
                let mode = manager
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .get_mermaid_rendering_mode();
                mode
            }),
            Some(theme()),
        );

        let mode = Rc::new_cyclic(move |weak: &std::rc::Weak<Self>| {
            let theme_controller = {
                let get_settings = Arc::clone(&settings);
                let show_error_weak = weak.clone();
                let changed_weak = weak.clone();
                let theme_ui: Rc<dyn ThemeUi> = Rc::new(TuiThemeUi::new(Rc::clone(&tui)));
                InteractiveThemeController::new(
                    theme_ui,
                    ThemeControllerOptions {
                        get_settings_manager: Rc::new(move || {
                            get_settings
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .clone()
                        }),
                        show_error: Rc::new(move |message: &str| {
                            if let Some(mode) = show_error_weak.upgrade() {
                                mode.show_error(message);
                            }
                        }),
                        on_changed: Rc::new(move || {
                            if let Some(mode) = changed_weak.upgrade() {
                                mode.update_editor_border_color();
                            }
                        }),
                        initial_theme_setting: options.initial_theme_setting.clone(),
                    },
                )
            };
            let auto_trust_on_reload_cwd = options.auto_trust_on_reload_cwd.clone();
            Self {
                runtime_host: RefCell::new(Some(runtime_host)),
                settings,
                tui,
                alt_screen: RefCell::new(alt_screen),
                main_screen: RefCell::new(main_screen),
                main_screen_render_state: RefCell::new(None),
                loaded_resources_container,
                chat_container,
                document_container,
                transcript_scroll_view: RefCell::new(None),
                fullscreen_layout_root: RefCell::new(None),
                pending_messages_container,
                status_container,
                default_editor,
                editor: RefCell::new(editor),
                editor_component_factory: RefCell::new(None),
                autocomplete_provider: RefCell::new(None),
                autocomplete_provider_wrappers: RefCell::new(Vec::new()),
                fd_path: RefCell::new(None),
                editor_container,
                active_selector_token: RefCell::new(None),
                active_selector_dispose: RefCell::new(None),
                footer,
                footer_container,
                footer_data_provider,
                keybindings,
                version: VERSION.to_owned(),
                is_initialized: Cell::new(false),
                active_status_indicator: RefCell::new(None),
                active_working_indicator_embedded: Cell::new(false),
                idle_status: Rc::new(IdleStatus),
                working_message: RefCell::new(None),
                working_visible: Cell::new(true),
                working_indicator_options: RefCell::new(None),
                hidden_thinking_label: RefCell::new(Self::DEFAULT_HIDDEN_THINKING_LABEL.to_owned()),
                last_sigint_time: Cell::new(0),
                last_escape_time: Cell::new(0),
                changelog_markdown: RefCell::new(None),
                startup_notices_shown: Cell::new(false),
                anthropic_subscription_warning_shown: Cell::new(false),
                last_status_spacer: RefCell::new(None),
                last_status_text: RefCell::new(None),
                expandable_rows: RefCell::new(Vec::new()),
                managed_tool_status_started: Cell::new(false),
                assistant_rows: RefCell::new(Vec::new()),
                streaming_component: RefCell::new(None),
                streaming_message: RefCell::new(None),
                pending_tools: RefCell::new(HashMap::new()),
                tool_output_expanded: Cell::new(false),
                hide_thinking_block: Cell::new(hide_thinking_block),
                output_pad: Cell::new(output_pad),
                mermaid_markdown_transformer,
                skill_commands: RefCell::new(HashMap::new()),
                signal_cleanup_handlers: RefCell::new(Vec::new()),
                is_bash_mode: Cell::new(false),
                bash_component: RefCell::new(None),
                pending_bash_components: RefCell::new(Vec::new()),
                auto_compaction_escape_handler: RefCell::new(None),
                retry_escape_handler: RefCell::new(None),
                compaction_queued_messages: RefCell::new(Vec::new()),
                shutdown_requested: Cell::new(false),
                is_shutting_down: Cell::new(false),
                extension_selector: RefCell::new(None),
                extension_input: RefCell::new(None),
                extension_editor: RefCell::new(None),
                extension_terminal_input_subscriptions: RefCell::new(Vec::new()),
                extension_widgets_above: RefCell::new(BTreeMap::new()),
                extension_widgets_below: RefCell::new(BTreeMap::new()),
                widget_container_above,
                widget_container_below,
                custom_footer: RefCell::new(None),
                header_container,
                built_in_header: RefCell::new(None),
                custom_header: RefCell::new(None),
                login_dialog: RefCell::new(None),
                options: RefCell::new(options),
                auto_trust_on_reload_cwd: RefCell::new(auto_trust_on_reload_cwd),
                theme_controller,
                unsubscribe: RefCell::new(Some(Box::new(unsubscribe))),
                event_sender,
                shutdown_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                command_tx,
                command_rx: RefCell::new(Some(command_rx)),
                event_rx: RefCell::new(Some(event_rx)),
                ui_requests: RefCell::new(Some(ui_requests)),
                ui_context,
            }
        });
        mode
    }

    /// The default working message.
    pub(crate) const DEFAULT_WORKING_MESSAGE: &'static str = "Working";

    /// The default hidden-thinking label.
    pub(crate) const DEFAULT_HIDDEN_THINKING_LABEL: &'static str = "Thinking...";

    /// The maximum widget lines that keep the viewport from overflowing.
    pub(crate) const MAX_WIDGET_LINES: usize = 10;

    /// The runtime host, taken for the duration of one replacement.
    pub(crate) fn take_runtime(&self) -> AgentSessionRuntime<S> {
        self.runtime_host
            .borrow_mut()
            .take()
            .expect("the runtime host is present outside a replacement")
    }

    /// Put the runtime host back after a replacement.
    pub(crate) fn put_runtime(&self, runtime: AgentSessionRuntime<S>) {
        *self.runtime_host.borrow_mut() = Some(runtime);
    }

    /// The current session.
    ///
    /// Public because the acceptance suites assert the session-side outcome
    /// of a frame action the way upstream's tests read the mocked session.
    ///
    /// # Panics
    ///
    /// During a runtime replacement, while the host is taken.
    #[must_use]
    pub fn session(&self) -> Arc<AgentSession<S>> {
        Arc::clone(
            self.runtime_host
                .borrow()
                .as_ref()
                .expect("the runtime host is present outside a replacement")
                .session(),
        )
    }

    /// The settings manager the current session shares.
    pub(crate) fn settings(&self) -> Arc<Mutex<SettingsManager<S>>> {
        self.settings
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Replace the settings slot after a session replacement.
    pub(crate) fn set_settings_manager(&self, manager: Arc<Mutex<SettingsManager<S>>>) {
        *self.settings.lock().unwrap_or_else(PoisonError::into_inner) = manager;
    }

    /// The current TUI.
    ///
    /// The acceptance suites read `mode`, focus, and children through it, the
    /// state upstream's tests reach through the frame's `ui` field.
    #[must_use]
    pub fn tui(&self) -> Rc<Tui> {
        Rc::clone(&self.tui)
    }

    /// The render request the mounted components use.
    pub(crate) fn render_request(&self) -> pi_tui::tui::RenderRequest {
        self.tui.render_request()
    }

    /// The current editor.
    #[must_use]
    pub fn editor(&self) -> Rc<dyn AppEditor> {
        Rc::clone(&self.editor.borrow())
    }

    /// The frame command sender.
    pub(crate) fn commands(&self) -> mpsc::UnboundedSender<FrameCommand> {
        self.command_tx.clone()
    }

    /// The current mode of the mounted TUI, in the settings' vocabulary.
    #[must_use]
    pub fn current_tui_mode(&self) -> TuiMode {
        self.tui.mode()
    }

    /// The transcript container, upstream's `chatContainer`.
    #[must_use]
    pub fn chat_container(&self) -> Rc<Container> {
        Rc::clone(&self.chat_container)
    }

    /// The startup resource-listing container, upstream's
    /// `loadedResourcesContainer`.
    #[must_use]
    pub fn loaded_resources_container(&self) -> Rc<Container> {
        Rc::clone(&self.loaded_resources_container)
    }

    /// The standalone status-indicator container, upstream's
    /// `statusContainer`.
    #[must_use]
    pub fn status_container(&self) -> Rc<Container> {
        Rc::clone(&self.status_container)
    }

    /// The active status indicator's kind, upstream's
    /// `activeStatusIndicator?.kind`.
    #[must_use]
    pub fn active_status_kind(
        &self,
    ) -> Option<crate::modes::interactive::components::status_indicator::StatusIndicatorKind> {
        self.active_status_indicator
            .borrow()
            .as_ref()
            .map(|indicator| indicator.kind())
    }

    /// The active generic extension selector, upstream's
    /// `extensionSelector` field.
    #[must_use]
    pub fn extension_selector(
        &self,
    ) -> Option<
        Rc<crate::modes::interactive::components::extension_selector::ExtensionSelectorComponent>,
    > {
        self.extension_selector.borrow().clone()
    }

    /// The extension UI context the session's extensions are bound with.
    #[must_use]
    pub fn ui_context(&self) -> Arc<extensions::InteractiveUiContext> {
        Arc::clone(&self.ui_context)
    }

    /// Drain every queued session event synchronously, the run loop's event
    /// half. The acceptance suites drive one event and apply it without
    /// starting the loop.
    pub fn drain_events(&self) {
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
    }

    /// Drain every queued extension UI request synchronously, the run loop's
    /// `drainEventsAndUI` half without the session-event half. The acceptance
    /// suites drive one wire call and apply it without starting the loop.
    pub fn drain_ui_requests(&self) {
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
}

/// Load the registered themes from the loader's handles, upstream's
/// `setRegisteredThemes(resourceLoader.getThemes().themes)`.
fn load_registered_themes<S: SettingsStorageKind>(
    session: &Arc<AgentSession<S>>,
) -> Vec<Arc<crate::modes::interactive::theme::theme::Theme>> {
    session
        .with_resource_loader(|loader| loader.get_themes().themes)
        .into_iter()
        .filter_map(|handle| handle.source_path)
        .filter_map(|path| {
            crate::modes::interactive::theme::theme::load_theme_from_path(
                std::path::Path::new(&path),
                None,
            )
            .ok()
        })
        .map(Arc::new)
        .collect()
}
