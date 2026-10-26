//! Settings selector, upstream's `settings-selector.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#206).
//!
//! The component renders a fully resolved settings snapshot and reports each
//! change through [`SettingsCallbacks`]; persistence and store access stay
//! with the interactive frame. The theme submenu carries its fixed and
//! automatic light/dark modes, the warnings submenu its per-warning toggles,
//! and the per-model thinking step its override and clear paths.
//!
//! Restatements: upstream's callback bag becomes [`SettingsCallbacks`];
//! `WarningSettings` is the typed projection of the settings store's
//! `warnings` object; and the model list is pre-indexed by
//! `provider/modelId` instead of the upstream `Map`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use pi_agent_core::types::ThinkingLevel;
use pi_ai::models::get_supported_thinking_levels;
use pi_ai::types::{Model, ModelThinkingLevel, Transport};
use pi_tui::components::scroll_view::ScrollViewScrollbar;
use pi_tui::components::select_list::{SelectItem, SelectListLayoutOptions};
use pi_tui::components::settings_list::{
    SettingItem, SettingsChangeCallback, SettingsList, SettingsListOptions, SettingsSubmenuDone,
    SettingsSubmenuFn,
};
use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::terminal_image::get_capabilities;
use pi_tui::tui::{Component, Container, TuiMode, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::key_display_text;
use super::settings_submenu::{
    SelectSubmenu, SteppedSubmenu, SteppedSubmenuOptions, SteppedSubmenuStep, SteppedSubmenuText,
};
use super::thinking_selector::level_details;
use crate::http_dispatcher::{HTTP_IDLE_TIMEOUT_CHOICES, format_http_idle_timeout_ms};
use crate::modes::interactive::theme::theme::{
    TerminalTheme, bold, get_settings_list_theme, parse_auto_theme_setting, theme,
};
use crate::settings_manager::{
    DefaultProjectTrust, FullscreenExitOutput, MermaidRenderingMode, QueueMode,
};

/// The model-picker column bounds, upstream's `MODEL_PICKER_LAYOUT`.
fn model_picker_layout() -> SelectListLayoutOptions {
    SelectListLayoutOptions {
        min_primary_column_width: Some(12),
        max_primary_column_width: Some(46),
        truncate_primary: None,
    }
}

/// The trust labels in cycle order, upstream's `DEFAULT_PROJECT_TRUST_LABELS`.
const DEFAULT_PROJECT_TRUST_LABELS: [(DefaultProjectTrust, &str); 3] = [
    (DefaultProjectTrust::Ask, "Ask"),
    (DefaultProjectTrust::Always, "Always trust"),
    (DefaultProjectTrust::Never, "Never trust"),
];

/// The automatic-theme sentinel value, upstream's `AUTOMATIC_THEME_VALUE`.
const AUTOMATIC_THEME_VALUE: &str = "/";

/// The clear-override sentinel value, upstream's `CLEAR_OVERRIDE_VALUE`.
const CLEAR_OVERRIDE_VALUE: &str = "__clear__";

/// The warning toggles, upstream's `WarningSettings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WarningSettings {
    /// Warn when Anthropic subscription auth may use paid extra usage;
    /// `true` when the setting is unset.
    pub anthropic_extra_usage: bool,
}

impl Default for WarningSettings {
    fn default() -> Self {
        Self {
            anthropic_extra_usage: true,
        }
    }
}

/// A callback carrying a model's provider id, model id, and new level.
pub type ModelThinkingLevelChangeFn = Rc<dyn Fn(&str, &str, ThinkingLevel)>;

/// A callback carrying a model's provider id and model id.
pub type ModelKeyCallback = Rc<dyn Fn(&str, &str)>;

/// An optional callback carrying one string value.
pub type OptionalStringCallback = Option<Rc<dyn Fn(&str)>>;

/// The settings snapshot the selector renders.
#[expect(
    clippy::struct_excessive_bools,
    reason = "the snapshot mirrors upstream's SettingsConfig field for field; each bool is a distinct on-disk setting"
)]
#[derive(Debug, Clone)]
pub struct SettingsConfig {
    /// Automatically compact context when it gets too large.
    pub auto_compact: bool,
    /// The settings default model key, `provider/modelId`.
    pub default_model: String,
    /// The session's current model.
    pub current_model: Option<Model>,
    /// Every model the per-model thinking step offers.
    pub available_default_models: Vec<Model>,
    /// Render images inline in the terminal.
    pub show_images: bool,
    /// Preferred inline image width in terminal cells.
    pub image_width_cells: i64,
    /// Resize large images before sending them.
    pub auto_resize_images: bool,
    /// Prevent images from being sent to providers.
    pub block_images: bool,
    /// Register skills as `/skill:name` commands.
    pub enable_skill_commands: bool,
    /// How steering messages queue while streaming.
    pub steering_mode: QueueMode,
    /// How follow-up messages queue until the agent stops.
    pub follow_up_mode: QueueMode,
    /// The preferred provider transport.
    pub transport: Transport,
    /// The HTTP idle timeout in milliseconds; `0` disables it.
    pub http_idle_timeout_ms: i64,
    /// The global thinking level.
    pub thinking_level: ThinkingLevel,
    /// The levels the session can cycle through.
    pub available_thinking_levels: Vec<ThinkingLevel>,
    /// Per-model thinking-level overrides, keyed by `provider/modelId`.
    pub model_thinking_levels: BTreeMap<String, ThinkingLevel>,
    /// The current theme setting, fixed name or `light/dark`.
    pub current_theme: String,
    /// The terminal's light or dark appearance.
    pub terminal_theme: TerminalTheme,
    /// Every loadable theme name.
    pub available_themes: Vec<String>,
    /// Hide thinking blocks in assistant responses.
    pub hide_thinking_block: bool,
    /// Mermaid rendering mode.
    pub mermaid_rendering_mode: MermaidRenderingMode,
    /// Show cache cost and provider recovery notices.
    pub show_cache_miss_notices: bool,
    /// Show condensed changelogs after updates.
    pub collapse_changelog: bool,
    /// Send an anonymous install telemetry ping after updates.
    pub enable_install_telemetry: bool,
    /// Action on a double Escape with an empty editor.
    pub double_escape_action: String,
    /// Default filter when opening `/tree`.
    pub tree_filter_mode: String,
    /// Show the terminal cursor while positioning it for IME support.
    pub show_hardware_cursor: bool,
    /// Horizontal padding for the input editor, 0–3.
    pub editor_padding_x: i64,
    /// Horizontal padding for messages and thinking, 0 or 1.
    pub output_pad: u8,
    /// Maximum visible autocomplete rows.
    pub autocomplete_max_visible: i64,
    /// Disable verbose printing at startup.
    pub quiet_startup: bool,
    /// Fallback behavior when nothing else decides project trust.
    pub default_project_trust: DefaultProjectTrust,
    /// Clear empty rows when content shrinks.
    pub clear_on_shrink: bool,
    /// Show OSC 9;4 progress indicators.
    pub show_terminal_progress: bool,
    /// Regular scrollback or fullscreen layout.
    pub tui_mode: TuiMode,
    /// Output printed when exiting fullscreen mode.
    pub fullscreen_exit_output: FullscreenExitOutput,
    /// Scrollbar behavior in fullscreen mode.
    pub fullscreen_scrollbar: ScrollViewScrollbar,
    /// Automatically copy selected text in fullscreen mode.
    pub fullscreen_copy_on_select: bool,
    /// The warning toggles.
    pub warnings: WarningSettings,
}

/// The settings change callbacks; persistence belongs to the host.
#[derive(Clone)]
pub struct SettingsCallbacks {
    /// Auto-compact toggled.
    pub on_auto_compact_change: Rc<dyn Fn(bool)>,
    /// Inline images toggled.
    pub on_show_images_change: Rc<dyn Fn(bool)>,
    /// Inline image width changed.
    pub on_image_width_cells_change: Rc<dyn Fn(i64)>,
    /// Image auto-resize toggled.
    pub on_auto_resize_images_change: Rc<dyn Fn(bool)>,
    /// Image blocking toggled.
    pub on_block_images_change: Rc<dyn Fn(bool)>,
    /// Skill commands toggled.
    pub on_enable_skill_commands_change: Rc<dyn Fn(bool)>,
    /// Steering queue mode changed.
    pub on_steering_mode_change: Rc<dyn Fn(QueueMode)>,
    /// Follow-up queue mode changed.
    pub on_follow_up_mode_change: Rc<dyn Fn(QueueMode)>,
    /// Preferred transport changed.
    pub on_transport_change: Rc<dyn Fn(Transport)>,
    /// HTTP idle timeout changed.
    pub on_http_idle_timeout_ms_change: Rc<dyn Fn(i64)>,
    /// A model's default thinking level changed.
    pub on_model_thinking_level_change: ModelThinkingLevelChangeFn,
    /// A model's thinking-level override was cleared.
    pub on_model_thinking_level_remove: ModelKeyCallback,
    /// The theme setting changed.
    pub on_theme_change: Rc<dyn Fn(&str)>,
    /// A theme is previewed while browsing; `None` leaves the theme alone.
    pub on_theme_preview: OptionalStringCallback,
    /// Thinking-block hiding toggled.
    pub on_hide_thinking_block_change: Rc<dyn Fn(bool)>,
    /// Mermaid rendering mode changed.
    pub on_mermaid_rendering_mode_change: Rc<dyn Fn(MermaidRenderingMode)>,
    /// Cache-miss notices toggled.
    pub on_show_cache_miss_notices_change: Rc<dyn Fn(bool)>,
    /// Changelog collapsing toggled.
    pub on_collapse_changelog_change: Rc<dyn Fn(bool)>,
    /// Install telemetry toggled.
    pub on_enable_install_telemetry_change: Rc<dyn Fn(bool)>,
    /// Double-escape action changed.
    pub on_double_escape_action_change: Rc<dyn Fn(&str)>,
    /// Tree filter mode changed.
    pub on_tree_filter_mode_change: Rc<dyn Fn(&str)>,
    /// Hardware cursor toggled.
    pub on_show_hardware_cursor_change: Rc<dyn Fn(bool)>,
    /// Editor padding changed.
    pub on_editor_padding_x_change: Rc<dyn Fn(i64)>,
    /// Output padding changed.
    pub on_output_pad_change: Rc<dyn Fn(u8)>,
    /// Autocomplete row limit changed.
    pub on_autocomplete_max_visible_change: Rc<dyn Fn(i64)>,
    /// Quiet startup toggled.
    pub on_quiet_startup_change: Rc<dyn Fn(bool)>,
    /// Default project trust changed.
    pub on_default_project_trust_change: Rc<dyn Fn(DefaultProjectTrust)>,
    /// Clear-on-shrink toggled.
    pub on_clear_on_shrink_change: Rc<dyn Fn(bool)>,
    /// Terminal progress indicators toggled.
    pub on_show_terminal_progress_change: Rc<dyn Fn(bool)>,
    /// TUI mode changed.
    pub on_tui_mode_change: Rc<dyn Fn(TuiMode)>,
    /// Fullscreen exit output changed.
    pub on_fullscreen_exit_output_change: Rc<dyn Fn(FullscreenExitOutput)>,
    /// Fullscreen scrollbar changed.
    pub on_fullscreen_scrollbar_change: Rc<dyn Fn(ScrollViewScrollbar)>,
    /// Fullscreen copy-on-select toggled.
    pub on_fullscreen_copy_on_select_change: Rc<dyn Fn(bool)>,
    /// Warning toggles changed.
    pub on_warnings_change: Rc<dyn Fn(WarningSettings)>,
    /// The selector was dismissed.
    pub on_cancel: Rc<dyn Fn()>,
}

impl std::fmt::Debug for SettingsCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsCallbacks").finish_non_exhaustive()
    }
}

fn model_setting_key(model: &Model) -> String {
    format!("{}/{}", model.provider.0, model.id)
}

fn model_display_label(model: &Model) -> String {
    format!("{} [{}]", model.id, model.provider.0)
}

fn model_thinking_overrides_summary(overrides: &BTreeMap<String, ThinkingLevel>) -> String {
    let count = overrides.len();
    if count == 0 {
        "none".to_string()
    } else {
        format!("{count} configured")
    }
}

fn model_item_label(model: &Model) -> String {
    format!(
        "{} {}",
        model.id,
        theme().fg("muted", &format!("[{}]", model.provider.0))
    )
}

fn theme_items(available_themes: &[String], current_theme: &str) -> Vec<SelectItem> {
    available_themes
        .iter()
        .map(|name| SelectItem {
            value: name.clone(),
            label: format!("{}{name}", if name == current_theme { "✓ " } else { "  " }),
            description: None,
        })
        .collect()
}

fn single_mode_theme_items(available_themes: &[String], current_theme: &str) -> Vec<SelectItem> {
    let mut items = vec![SelectItem {
        value: AUTOMATIC_THEME_VALUE.to_string(),
        label: "  Automatic".to_string(),
        description: Some("Use separate themes for light and dark terminal appearance".to_string()),
    }];
    items.extend(theme_items(available_themes, current_theme));
    items
}

fn preferred_theme(available_themes: &[String], preferred: Option<&str>, fallback: &str) -> String {
    if let Some(preferred) = preferred
        && available_themes.iter().any(|name| name == preferred)
    {
        return preferred.to_string();
    }
    if available_themes.iter().any(|name| name == fallback) {
        return fallback.to_string();
    }
    available_themes
        .first()
        .cloned()
        .unwrap_or_else(|| fallback.to_string())
}

/// Upstream's `defaultAutomaticThemes`: the parsed pair, else the fixed
/// current theme, else `dark`, for both appearances.
fn default_automatic_themes(
    current_theme_setting: &str,
    available_themes: &[String],
) -> (String, String) {
    if let Some(auto_theme) = parse_auto_theme_setting(Some(current_theme_setting)) {
        return (auto_theme.light_theme, auto_theme.dark_theme);
    }
    let current_fixed_theme =
        (!current_theme_setting.contains('/')).then_some(current_theme_setting);
    let theme_name = preferred_theme(available_themes, current_fixed_theme, "dark");
    (theme_name.clone(), theme_name)
}

fn thinking_level_from_value(value: &str) -> Option<ThinkingLevel> {
    match value {
        "off" => Some(ThinkingLevel::Off),
        "minimal" => Some(ThinkingLevel::Minimal),
        "low" => Some(ThinkingLevel::Low),
        "medium" => Some(ThinkingLevel::Medium),
        "high" => Some(ThinkingLevel::High),
        "xhigh" => Some(ThinkingLevel::Xhigh),
        "max" => Some(ThinkingLevel::Max),
        _ => None,
    }
}

const fn model_thinking_level(level: ModelThinkingLevel) -> ThinkingLevel {
    match level {
        ModelThinkingLevel::Off => ThinkingLevel::Off,
        ModelThinkingLevel::Minimal => ThinkingLevel::Minimal,
        ModelThinkingLevel::Low => ThinkingLevel::Low,
        ModelThinkingLevel::Medium => ThinkingLevel::Medium,
        ModelThinkingLevel::High => ThinkingLevel::High,
        ModelThinkingLevel::Xhigh => ThinkingLevel::Xhigh,
        ModelThinkingLevel::Max => ThinkingLevel::Max,
    }
}

fn trust_label(trust: DefaultProjectTrust) -> &'static str {
    DEFAULT_PROJECT_TRUST_LABELS
        .iter()
        .find_map(|(value, label)| (*value == trust).then_some(*label))
        .unwrap_or("Ask")
}

fn trust_from_label(label: &str) -> Option<DefaultProjectTrust> {
    DEFAULT_PROJECT_TRUST_LABELS
        .iter()
        .find_map(|(value, candidate)| (*candidate == label).then_some(*value))
}

/// The per-warning toggles, upstream `WarningSettingsSubmenu`.
struct WarningSettingsSubmenu {
    settings_list: Rc<SettingsList>,
}

impl WarningSettingsSubmenu {
    fn new(
        warnings: WarningSettings,
        on_change: Rc<dyn Fn(WarningSettings)>,
        done: SettingsSubmenuDone,
    ) -> Self {
        let state = Rc::new(RefCell::new(warnings));
        let items = vec![SettingItem {
            id: "anthropic-extra-usage".to_string(),
            label: "Anthropic extra usage".to_string(),
            description: Some(
                "Warn when Anthropic subscription auth may use paid extra usage".to_string(),
            ),
            current_value: if warnings.anthropic_extra_usage {
                "true".to_string()
            } else {
                "false".to_string()
            },
            values: Some(vec!["true".to_string(), "false".to_string()]),
            submenu: None,
        }];
        let state_for_change = Rc::clone(&state);
        let settings_list = Rc::new(SettingsList::new(
            items,
            1,
            get_settings_list_theme(),
            Rc::new(move |_id, new_value| {
                let mut warnings = *state_for_change.borrow();
                warnings.anthropic_extra_usage = new_value == "true";
                *state_for_change.borrow_mut() = warnings;
                on_change(warnings);
            }),
            Rc::new(move || done(None, None)),
        ));
        Self { settings_list }
    }
}

impl Component for WarningSettingsSubmenu {
    fn render(&self, width: usize) -> Vec<String> {
        self.settings_list.render(width)
    }

    fn handle_input(&self, data: &str) {
        self.settings_list.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.settings_list.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        let settings_list: Rc<dyn Component> = self.settings_list.clone();
        vec![settings_list]
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.settings_list.handle_mouse(event)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemeMode {
    Single,
    Automatic,
}

#[derive(Debug, Clone)]
struct ThemeState {
    mode: ThemeMode,
    single_theme: String,
    light_theme: String,
    dark_theme: String,
}

/// Shared state for the theme submenu, upstream `ThemeSubmenu`'s fields.
struct ThemeShared {
    container: Rc<Container>,
    input_component: RefCell<Option<Rc<dyn Component>>>,
    callbacks: SettingsCallbacks,
    available_themes: Vec<String>,
    terminal_theme: TerminalTheme,
    on_done: SettingsSubmenuDone,
    original_theme_setting: String,
    state: RefCell<ThemeState>,
}

/// Applies one theme selection to the shared state.
type ThemeUpdate = Rc<dyn Fn(&ThemeShared, &str)>;

impl ThemeShared {
    fn preview(&self, theme_setting: &str) {
        if let Some(on_theme_preview) = &self.callbacks.on_theme_preview {
            on_theme_preview(theme_setting);
        }
    }

    fn get_active_automatic_theme(&self) -> String {
        let state = self.state.borrow();
        if self.terminal_theme == TerminalTheme::Light {
            state.light_theme.clone()
        } else {
            state.dark_theme.clone()
        }
    }

    fn get_automatic_theme_setting(&self) -> String {
        let state = self.state.borrow();
        format!("{}/{}", state.light_theme, state.dark_theme)
    }

    fn get_theme_setting(&self) -> String {
        if self.state.borrow().mode == ThemeMode::Automatic {
            self.get_automatic_theme_setting()
        } else {
            self.state.borrow().single_theme.clone()
        }
    }

    fn apply(&self, theme_setting: &str) {
        (self.on_done)(Some(theme_setting.to_string()), None);
    }

    fn cancel(&self) {
        self.preview(&self.original_theme_setting);
        (self.on_done)(None, None);
    }

    fn set_content(&self, render: Rc<dyn Component>, input: Rc<dyn Component>) {
        self.container.clear();
        self.container.add_child(render);
        *self.input_component.borrow_mut() = Some(input);
    }

    fn show_single_menu(shared: &Rc<Self>) {
        let single_theme = {
            let mut state = shared.state.borrow_mut();
            state.mode = ThemeMode::Single;
            state.single_theme.clone()
        };

        let select_shared = Rc::downgrade(shared);
        let on_select: Rc<dyn Fn(&str)> = Rc::new(move |value: &str| {
            let Some(shared) = select_shared.upgrade() else {
                return;
            };
            if value == AUTOMATIC_THEME_VALUE {
                shared.state.borrow_mut().mode = ThemeMode::Automatic;
                let setting = shared.get_theme_setting();
                shared.preview(&setting);
                Self::show_automatic_menu(&shared);
                return;
            }
            shared.state.borrow_mut().single_theme = value.to_string();
            shared.apply(value);
        });
        let cancel_shared = Rc::downgrade(shared);
        let on_cancel: Rc<dyn Fn()> = Rc::new(move || {
            if let Some(shared) = cancel_shared.upgrade() {
                shared.cancel();
            }
        });
        let change_shared = Rc::downgrade(shared);
        let on_selection_change: Rc<dyn Fn(&str)> = Rc::new(move |value: &str| {
            if let Some(shared) = change_shared.upgrade() {
                let setting = if value == AUTOMATIC_THEME_VALUE {
                    shared.get_automatic_theme_setting()
                } else {
                    value.to_string()
                };
                shared.preview(&setting);
            }
        });

        let menu = Rc::new(SelectSubmenu::new(
            "Theme",
            "Select a theme, or choose Automatic to follow terminal appearance.",
            single_mode_theme_items(&shared.available_themes, &single_theme),
            &single_theme,
            on_select,
            on_cancel,
            Some(on_selection_change),
            None,
        ));
        let menu: Rc<dyn Component> = menu;
        shared.set_content(Rc::clone(&menu), menu);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's single automatic-theme layout method; splitting it would detach the port from the source"
    )]
    fn show_automatic_menu(shared: &Rc<Self>) {
        shared.state.borrow_mut().mode = ThemeMode::Automatic;
        let content = Rc::new(Container::default());
        content.add_child(Rc::new(Text::with_padding(
            bold(&theme().fg("accent", "Automatic Theme")),
            0,
            0,
        )));
        content.add_child(Rc::new(Spacer::new()));
        content.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "muted",
                "Choose themes for terminal light and dark appearance.",
            ),
            0,
            0,
        )));
        content.add_child(Rc::new(Text::with_padding(
            theme().fg("muted", "Light/dark detection requires terminal support."),
            0,
            0,
        )));
        content.add_child(Rc::new(Spacer::new()));

        let light_submenu: SettingsSubmenuFn = {
            let weak = Rc::downgrade(shared);
            Rc::new(move |current_value, done| {
                let Some(shared) = weak.upgrade() else {
                    return Rc::new(Spacer::with_lines(0));
                };
                let update: ThemeUpdate = Rc::new(|shared, value| {
                    shared.state.borrow_mut().light_theme = value.to_string();
                    let setting = shared.get_theme_setting();
                    shared.preview(&setting);
                });
                Rc::new(Self::create_theme_select(
                    &shared,
                    "Light Theme",
                    "Select the theme to use for light terminal appearance",
                    current_value,
                    &done,
                    update,
                ))
            })
        };
        let dark_submenu: SettingsSubmenuFn = {
            let weak = Rc::downgrade(shared);
            Rc::new(move |current_value, done| {
                let Some(shared) = weak.upgrade() else {
                    return Rc::new(Spacer::with_lines(0));
                };
                let update: ThemeUpdate = Rc::new(|shared, value| {
                    shared.state.borrow_mut().dark_theme = value.to_string();
                    let setting = shared.get_theme_setting();
                    shared.preview(&setting);
                });
                Rc::new(Self::create_theme_select(
                    &shared,
                    "Dark Theme",
                    "Select the theme to use for dark terminal appearance",
                    current_value,
                    &done,
                    update,
                ))
            })
        };

        let light_theme = shared.state.borrow().light_theme.clone();
        let dark_theme = shared.state.borrow().dark_theme.clone();
        let items = vec![
            SettingItem {
                id: "light-theme".to_string(),
                label: "Light theme".to_string(),
                description: Some(
                    "Theme to use in automatic mode when the terminal is light".to_string(),
                ),
                current_value: light_theme,
                values: None,
                submenu: Some(light_submenu),
            },
            SettingItem {
                id: "dark-theme".to_string(),
                label: "Dark theme".to_string(),
                description: Some(
                    "Theme to use in automatic mode when the terminal is dark".to_string(),
                ),
                current_value: dark_theme,
                values: None,
                submenu: Some(dark_submenu),
            },
            SettingItem {
                id: "apply".to_string(),
                label: "Apply".to_string(),
                description: Some("Save and go back".to_string()),
                current_value: "save and go back".to_string(),
                values: Some(vec!["save and go back".to_string()]),
                submenu: None,
            },
            SettingItem {
                id: "single-mode".to_string(),
                label: "Change mode".to_string(),
                description: Some("Switch to one theme for light and dark".to_string()),
                current_value: "switch to single theme".to_string(),
                values: Some(vec!["switch to single theme".to_string()]),
                submenu: None,
            },
        ];

        let change_weak = Rc::downgrade(shared);
        let on_change: SettingsChangeCallback = Rc::new(move |id, _value| {
            let Some(shared) = change_weak.upgrade() else {
                return;
            };
            match id {
                "single-mode" => {
                    let active = shared.get_active_automatic_theme();
                    {
                        let mut state = shared.state.borrow_mut();
                        state.mode = ThemeMode::Single;
                        state.single_theme.clone_from(&active);
                    }
                    shared.preview(&active);
                    Self::show_single_menu(&shared);
                }
                "apply" => {
                    let setting = shared.get_automatic_theme_setting();
                    shared.apply(&setting);
                }
                _ => {}
            }
        });
        let cancel_weak = Rc::downgrade(shared);
        let on_cancel: Rc<dyn Fn()> = Rc::new(move || {
            if let Some(shared) = cancel_weak.upgrade() {
                shared.cancel();
            }
        });
        let settings_list = Rc::new(SettingsList::new(
            items,
            4,
            get_settings_list_theme(),
            on_change,
            on_cancel,
        ));
        let settings_component: Rc<dyn Component> = settings_list.clone();
        content.add_child(settings_component);
        let content_component: Rc<dyn Component> = content;
        shared.set_content(content_component, settings_list);
    }

    fn create_theme_select(
        shared: &Rc<Self>,
        title: &str,
        description: &str,
        current_value: &str,
        done: &SettingsSubmenuDone,
        update: ThemeUpdate,
    ) -> SelectSubmenu {
        let select_weak = Rc::downgrade(shared);
        let done_select = Rc::clone(done);
        let on_select: Rc<dyn Fn(&str)> = Rc::new(move |value: &str| {
            if let Some(shared) = select_weak.upgrade() {
                update(&shared, value);
                done_select(Some(value.to_string()), None);
            }
        });
        let cancel_weak = Rc::downgrade(shared);
        let done_cancel = Rc::clone(done);
        let on_cancel: Rc<dyn Fn()> = Rc::new(move || {
            if let Some(shared) = cancel_weak.upgrade() {
                let setting = shared.get_theme_setting();
                shared.preview(&setting);
            }
            done_cancel(None, None);
        });
        let change_weak = Rc::downgrade(shared);
        let on_selection_change: Rc<dyn Fn(&str)> = Rc::new(move |value: &str| {
            if let Some(shared) = change_weak.upgrade() {
                shared.preview(value);
            }
        });
        SelectSubmenu::new(
            title,
            description,
            theme_items(&shared.available_themes, current_value),
            current_value,
            on_select,
            on_cancel,
            Some(on_selection_change),
            None,
        )
    }
}

/// The theme submenu in fixed and automatic modes, upstream `ThemeSubmenu`.
pub struct ThemeSubmenu {
    shared: Rc<ThemeShared>,
}

impl std::fmt::Debug for ThemeSubmenu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThemeSubmenu")
            .field("state", &self.shared.state.borrow())
            .finish_non_exhaustive()
    }
}

impl ThemeSubmenu {
    /// Opens on the current setting's mode, upstream's `parseAutoThemeSetting`
    /// branch.
    #[must_use]
    pub fn new(
        current_theme_setting: &str,
        terminal_theme: TerminalTheme,
        available_themes: Vec<String>,
        callbacks: SettingsCallbacks,
        done: SettingsSubmenuDone,
    ) -> Self {
        let auto_theme = parse_auto_theme_setting(Some(current_theme_setting));
        let (light_theme, dark_theme) =
            default_automatic_themes(current_theme_setting, &available_themes);
        let mode = if auto_theme.is_some() {
            ThemeMode::Automatic
        } else {
            ThemeMode::Single
        };
        let fixed_theme = if auto_theme.is_some() || current_theme_setting.contains('/') {
            None
        } else {
            Some(current_theme_setting)
        };
        let shared = Rc::new(ThemeShared {
            container: Rc::new(Container::default()),
            input_component: RefCell::new(None),
            callbacks,
            available_themes,
            terminal_theme,
            on_done: done,
            original_theme_setting: current_theme_setting.to_string(),
            state: RefCell::new(ThemeState {
                mode,
                single_theme: String::new(),
                light_theme,
                dark_theme,
            }),
        });
        let automatic_theme =
            (mode == ThemeMode::Automatic).then(|| shared.get_active_automatic_theme());
        let single_theme = preferred_theme(
            &shared.available_themes,
            fixed_theme.or(automatic_theme.as_deref()),
            "dark",
        );
        shared.state.borrow_mut().single_theme = single_theme;

        if mode == ThemeMode::Automatic {
            ThemeShared::show_automatic_menu(&shared);
        } else {
            ThemeShared::show_single_menu(&shared);
        }
        Self { shared }
    }
}

impl Component for ThemeSubmenu {
    fn render(&self, width: usize) -> Vec<String> {
        self.shared.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let input_component = self.shared.input_component.borrow().clone();
        if let Some(input_component) = input_component {
            input_component.handle_input(data);
        }
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.shared.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.shared.container.children()
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.shared.container.handle_mouse(event)
    }
}

fn value_setting(
    id: &str,
    label: &str,
    description: &str,
    current_value: String,
    values: &[&str],
) -> SettingItem {
    SettingItem {
        id: id.to_string(),
        label: label.to_string(),
        description: Some(description.to_string()),
        current_value,
        values: Some(values.iter().map(ToString::to_string).collect()),
        submenu: None,
    }
}

const fn transport_value(transport: Transport) -> &'static str {
    match transport {
        Transport::Sse => "sse",
        Transport::Websocket => "websocket",
        Transport::WebsocketCached => "websocket-cached",
        Transport::Auto => "auto",
    }
}

/// The selector over the resolved settings snapshot, upstream
/// `SettingsSelectorComponent`.
pub struct SettingsSelectorComponent {
    container: Container,
    settings_list: Rc<SettingsList>,
}

impl std::fmt::Debug for SettingsSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsSelectorComponent")
            .finish_non_exhaustive()
    }
}

impl SettingsSelectorComponent {
    /// Builds every settings row; capability-gated image rows follow the
    /// terminal's detected image protocol.
    #[must_use]
    #[expect(
        clippy::too_many_lines,
        reason = "mirrors upstream's single constructor branch for branch; splitting it would detach the port from the method it mirrors"
    )]
    pub fn new(config: &SettingsConfig, callbacks: SettingsCallbacks) -> Self {
        let supports_images = get_capabilities().images.is_some();
        let follow_up_key = key_display_text("app.message.followUp");
        let cycle_thinking_key = key_display_text("app.thinking.cycle");
        let current_warnings = Rc::new(RefCell::new(config.warnings));
        let current_model_thinking_levels =
            Rc::new(RefCell::new(config.model_thinking_levels.clone()));
        let default_model_by_value: BTreeMap<String, Model> = config
            .available_default_models
            .iter()
            .cloned()
            .map(|model| (model_setting_key(&model), model))
            .collect();
        let current_default_model_key = default_model_by_value
            .contains_key(&config.default_model)
            .then(|| config.default_model.clone());
        let current_model_key = config.current_model.as_ref().map(model_setting_key);

        let warnings_submenu: SettingsSubmenuFn = {
            let warnings = Rc::clone(&current_warnings);
            let on_change = Rc::clone(&callbacks.on_warnings_change);
            Rc::new(move |_current_value, done| {
                let current = *warnings.borrow();
                let state = Rc::clone(&warnings);
                let change = Rc::clone(&on_change);
                Rc::new(WarningSettingsSubmenu::new(
                    current,
                    Rc::new(move |next| {
                        *state.borrow_mut() = next;
                        (change)(next);
                    }),
                    done,
                ))
            })
        };

        let model_thinking_submenu: SettingsSubmenuFn = {
            let levels = Rc::clone(&current_model_thinking_levels);
            let on_change = Rc::clone(&callbacks.on_model_thinking_level_change);
            let on_remove = Rc::clone(&callbacks.on_model_thinking_level_remove);
            let thinking_level = config.thinking_level;
            let models = config.available_default_models.clone();
            let models_by_key = default_model_by_value;
            let current_key = current_model_key;
            let default_key = current_default_model_key;
            Rc::new(move |_current_value, done| {
                let model_step_models = models.clone();
                let model_step_levels = Rc::clone(&levels);
                let model_step_current = current_key.clone();
                let model_step_default = default_key.clone();
                let model_preselect_current = model_step_current.clone();
                let model_preselect_default = model_step_default.clone();
                let model_step = SteppedSubmenuStep {
                    key: "model".to_string(),
                    title: SteppedSubmenuText::Static("Per-Model Thinking Level".to_string()),
                    description: SteppedSubmenuText::Static(
                        "Select a model to configure".to_string(),
                    ),
                    options: Rc::new(move |_context| {
                        let mut sorted = model_step_models.clone();
                        sorted.sort_by(|a, b| {
                            let a_key = model_setting_key(a);
                            let b_key = model_setting_key(b);
                            if Some(&a_key) == model_step_current.as_ref() {
                                return std::cmp::Ordering::Less;
                            }
                            if Some(&b_key) == model_step_current.as_ref() {
                                return std::cmp::Ordering::Greater;
                            }
                            if Some(&a_key) == model_step_default.as_ref() {
                                return std::cmp::Ordering::Less;
                            }
                            if Some(&b_key) == model_step_default.as_ref() {
                                return std::cmp::Ordering::Greater;
                            }
                            a.provider.0.cmp(&b.provider.0)
                        });
                        let mut items: Vec<SelectItem> = sorted
                            .iter()
                            .map(|model| {
                                let key = model_setting_key(model);
                                let override_level = model_step_levels
                                    .borrow()
                                    .get(&key)
                                    .map(|level| level_details(*level).0.to_string());
                                SelectItem {
                                    value: key,
                                    label: model_item_label(model),
                                    description: override_level,
                                }
                            })
                            .collect();
                        if items.is_empty() {
                            items.push(SelectItem {
                                value: "__none__".to_string(),
                                label: "No models available".to_string(),
                                description: Some(
                                    "Log in to a provider or configure an API key first"
                                        .to_string(),
                                ),
                            });
                        }
                        items
                    }),
                    preselect: Some(Rc::new(move |_context| {
                        model_preselect_current
                            .clone()
                            .or_else(|| model_preselect_default.clone())
                    })),
                    searchable: true,
                    layout: Some(model_picker_layout()),
                };
                let level_models = models_by_key.clone();
                let level_models_for_options = models_by_key.clone();
                let level_overrides = Rc::clone(&levels);
                let level_overrides_for_options = Rc::clone(&level_overrides);
                let level_step = SteppedSubmenuStep {
                    key: "level".to_string(),
                    title: SteppedSubmenuText::Dynamic(Rc::new(move |context| {
                        context
                            .get("model")
                            .and_then(|key| level_models.get(key))
                            .map_or_else(
                                || {
                                    format!(
                                        "Thinking Level for {}",
                                        context.get("model").map_or("", String::as_str)
                                    )
                                },
                                |model| {
                                    format!("Thinking Level for {}", model_display_label(model))
                                },
                            )
                    })),
                    description: SteppedSubmenuText::Static(
                        "Select default thinking level for this model".to_string(),
                    ),
                    options: Rc::new(move |context| {
                        let Some(model) = context
                            .get("model")
                            .and_then(|key| level_models_for_options.get(key))
                        else {
                            return Vec::new();
                        };
                        let levels = if model.reasoning {
                            get_supported_thinking_levels(model)
                                .into_iter()
                                .map(model_thinking_level)
                                .collect()
                        } else {
                            vec![ThinkingLevel::Off]
                        };
                        let active = context
                            .get("model")
                            .and_then(|key| level_overrides_for_options.borrow().get(key).copied());
                        let mut items: Vec<SelectItem> = levels
                            .iter()
                            .map(|level| {
                                let (value, description) = level_details(*level);
                                SelectItem {
                                    value: value.to_string(),
                                    label: format!(
                                        "{}{value}",
                                        if Some(*level) == active { "✓ " } else { "  " }
                                    ),
                                    description: Some(description.to_string()),
                                }
                            })
                            .collect();
                        if active.is_some() {
                            items.push(SelectItem {
                                value: CLEAR_OVERRIDE_VALUE.to_string(),
                                label: "  (clear override)".to_string(),
                                description: Some(format!(
                                    "Revert to global default ({})",
                                    level_details(thinking_level).0
                                )),
                            });
                        }
                        items
                    }),
                    preselect: Some(Rc::new(move |context| {
                        context.get("model").and_then(|key| {
                            level_overrides
                                .borrow()
                                .get(key)
                                .map(|level| level_details(*level).0.to_string())
                        })
                    })),
                    searchable: false,
                    layout: None,
                };

                let completion_models = models_by_key.clone();
                let completion_levels = Rc::clone(&levels);
                let completion_change = Rc::clone(&on_change);
                let completion_remove = Rc::clone(&on_remove);
                let on_complete = Rc::new(move |selections: BTreeMap<String, String>| {
                    let Some(model) = selections
                        .get("model")
                        .and_then(|key| completion_models.get(key))
                    else {
                        return;
                    };
                    let Some(level_value) = selections.get("level") else {
                        return;
                    };
                    let key = model_setting_key(model);
                    if level_value == CLEAR_OVERRIDE_VALUE {
                        (completion_remove)(&model.provider.0, &model.id);
                        completion_levels.borrow_mut().remove(&key);
                    } else if let Some(level) = thinking_level_from_value(level_value) {
                        (completion_change)(&model.provider.0, &model.id, level);
                        completion_levels.borrow_mut().insert(key, level);
                    }
                });
                let summary_levels = Rc::clone(&levels);
                let on_cancel = Rc::new(move || {
                    let summary = model_thinking_overrides_summary(&summary_levels.borrow());
                    done(Some(summary), None);
                });
                Rc::new(SteppedSubmenu::new(
                    vec![model_step, level_step],
                    on_complete,
                    on_cancel,
                    SteppedSubmenuOptions {
                        loop_: true,
                        ..SteppedSubmenuOptions::default()
                    },
                ))
            })
        };

        let theme_submenu: SettingsSubmenuFn = {
            let theme_callbacks = callbacks.clone();
            let available_themes = config.available_themes.clone();
            let terminal_theme = config.terminal_theme;
            Rc::new(move |current_value, done| {
                Rc::new(ThemeSubmenu::new(
                    current_value,
                    terminal_theme,
                    available_themes.clone(),
                    theme_callbacks.clone(),
                    done,
                ))
            })
        };

        let http_idle_timeout_values: Vec<&str> = HTTP_IDLE_TIMEOUT_CHOICES
            .iter()
            .map(|choice| choice.label)
            .collect();
        let mut items: Vec<SettingItem> = vec![value_setting(
            "autocompact",
            "Auto-compact",
            "Automatically compact context when it gets too large",
            config.auto_compact.to_string(),
            &["true", "false"],
        )];
        if supports_images {
            items.push(value_setting(
                "show-images",
                "Show images",
                "Render images inline in terminal",
                config.show_images.to_string(),
                &["true", "false"],
            ));
            items.push(value_setting(
                "image-width-cells",
                "Image width",
                "Preferred inline image width in terminal cells",
                config.image_width_cells.to_string(),
                &["60", "80", "120"],
            ));
        }
        items.push(value_setting(
            "auto-resize-images",
            "Auto-resize images",
            "Resize large images to 2000x2000 max for better model compatibility",
            config.auto_resize_images.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "block-images",
            "Block images",
            "Prevent images from being sent to LLM providers",
            config.block_images.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "skill-commands",
            "Skill commands",
            "Register skills as /skill:name commands",
            config.enable_skill_commands.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "show-hardware-cursor",
            "Show hardware cursor",
            "Show the terminal cursor while still positioning it for IME support",
            config.show_hardware_cursor.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "editor-padding",
            "Editor padding",
            "Horizontal padding for input editor (0-3)",
            config.editor_padding_x.to_string(),
            &["0", "1", "2", "3"],
        ));
        items.push(value_setting(
            "output-padding",
            "Output padding",
            "Horizontal padding for user messages, assistant messages, and thinking",
            config.output_pad.to_string(),
            &["0", "1"],
        ));
        items.push(value_setting(
            "autocomplete-max-visible",
            "Autocomplete max items",
            "Max visible items in autocomplete dropdown (3-20)",
            config.autocomplete_max_visible.to_string(),
            &["3", "5", "7", "10", "15", "20"],
        ));
        items.push(value_setting(
            "clear-on-shrink",
            "Clear on shrink",
            "Clear empty rows when content shrinks (may cause flicker)",
            config.clear_on_shrink.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "terminal-progress",
            "Terminal progress",
            "Show OSC 9;4 progress indicators in the terminal tab bar",
            config.show_terminal_progress.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "steering-mode",
            "Steering mode",
            "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once.",
            config.steering_mode.as_str().to_string(),
            &["one-at-a-time", "all"],
        ));
        items.push(value_setting(
            "follow-up-mode",
            "Follow-up mode",
            &format!(
                "{follow_up_key} queues follow-up messages until agent stops. 'one-at-a-time': deliver one, wait for response. 'all': deliver all at once."
            ),
            config.follow_up_mode.as_str().to_string(),
            &["one-at-a-time", "all"],
        ));
        items.push(value_setting(
            "transport",
            "Transport",
            "Preferred transport for providers that support multiple transports",
            transport_value(config.transport).to_string(),
            &["sse", "websocket", "websocket-cached", "auto"],
        ));
        items.push(value_setting(
            "http-idle-timeout",
            "HTTP idle timeout",
            "Maximum idle gap while waiting for HTTP headers or body chunks. Disable for local models that pause longer than five minutes.",
            format_http_idle_timeout_ms(config.http_idle_timeout_ms),
            &http_idle_timeout_values,
        ));
        items.push(value_setting(
            "hide-thinking",
            "Hide thinking",
            "Hide thinking blocks in assistant responses",
            config.hide_thinking_block.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "mermaid-rendering",
            "Mermaid diagrams",
            "Render Mermaid code blocks as Unicode diagrams",
            match config.mermaid_rendering_mode {
                MermaidRenderingMode::Off => "off",
                MermaidRenderingMode::Final => "final",
                MermaidRenderingMode::Streaming => "streaming",
            }
            .to_string(),
            &["off", "final", "streaming"],
        ));
        items.push(value_setting(
            "cache-miss-notices",
            "Cache miss notices",
            "Show transcript notices for cache costs and provider recovery diagnostics",
            config.show_cache_miss_notices.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "collapse-changelog",
            "Collapse changelog",
            "Show condensed changelog after updates",
            config.collapse_changelog.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "quiet-startup",
            "Quiet startup",
            "Disable verbose printing at startup",
            config.quiet_startup.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "install-telemetry",
            "Install telemetry",
            "Send an anonymous version/update ping after changelog-detected updates",
            config.enable_install_telemetry.to_string(),
            &["true", "false"],
        ));
        items.push(value_setting(
            "default-project-trust",
            "Default project trust",
            "Fallback behavior when no extension or saved trust decision decides project trust",
            trust_label(config.default_project_trust).to_string(),
            &["Ask", "Always trust", "Never trust"],
        ));
        items.push(value_setting(
            "double-escape-action",
            "Double-escape action",
            "Action when pressing Escape twice with empty editor",
            config.double_escape_action.clone(),
            &["tree", "fork", "none"],
        ));
        items.push(value_setting(
            "tree-filter-mode",
            "Tree filter mode",
            "Default filter when opening /tree",
            config.tree_filter_mode.clone(),
            &["default", "no-tools", "user-only", "labeled-only", "all"],
        ));
        items.push(SettingItem {
            id: "warnings".to_string(),
            label: "Warnings".to_string(),
            description: Some("Enable or disable individual warnings".to_string()),
            current_value: "configure".to_string(),
            values: None,
            submenu: Some(warnings_submenu),
        });
        items.push(SettingItem {
            id: "model-thinking".to_string(),
            label: "Default thinking level per model".to_string(),
            description: Some(format!(
                "Override the default thinking level for specific models. {cycle_thinking_key} cycles in-session."
            )),
            current_value: model_thinking_overrides_summary(
                &current_model_thinking_levels.borrow(),
            ),
            values: None,
            submenu: Some(model_thinking_submenu),
        });
        items.push(value_setting(
            "tui-mode",
            "TUI mode",
            "Interface layout; fullscreen mode is experimental",
            match config.tui_mode {
                TuiMode::Regular => "regular",
                TuiMode::Fullscreen => "fullscreen",
            }
            .to_string(),
            &["regular", "fullscreen"],
        ));
        items.push(value_setting(
            "fullscreen-exit-output",
            "Fullscreen exit output",
            "Print the transcript or only a session resume hint when exiting fullscreen mode",
            match config.fullscreen_exit_output {
                FullscreenExitOutput::Transcript => "transcript",
                FullscreenExitOutput::ResumeHint => "resume-hint",
            }
            .to_string(),
            &["transcript", "resume-hint"],
        ));
        items.push(value_setting(
            "fullscreen-scrollbar",
            "Fullscreen scrollbar",
            "Scrollbar behavior in fullscreen mode; has no effect in regular mode",
            match config.fullscreen_scrollbar {
                ScrollViewScrollbar::Auto => "auto",
                ScrollViewScrollbar::Always => "always",
                ScrollViewScrollbar::Hidden => "hidden",
            }
            .to_string(),
            &["auto", "always", "hidden"],
        ));
        items.push(value_setting(
            "fullscreen-copy-on-select",
            "Fullscreen copy on select",
            "Automatically copy selected text in fullscreen mode; disable to copy selections with Ctrl+X",
            config.fullscreen_copy_on_select.to_string(),
            &["true", "false"],
        ));
        items.push(SettingItem {
            id: "theme".to_string(),
            label: "Theme".to_string(),
            description: Some("Color theme for the interface".to_string()),
            current_value: config.current_theme.clone(),
            values: None,
            submenu: Some(theme_submenu),
        });

        let on_cancel = Rc::clone(&callbacks.on_cancel);
        let on_change: SettingsChangeCallback = Rc::new(move |id, new_value| match id {
            "autocompact" => (callbacks.on_auto_compact_change)(new_value == "true"),
            "show-images" => (callbacks.on_show_images_change)(new_value == "true"),
            "image-width-cells" => {
                (callbacks.on_image_width_cells_change)(new_value.parse().unwrap_or_default());
            }
            "auto-resize-images" => {
                (callbacks.on_auto_resize_images_change)(new_value == "true");
            }
            "block-images" => (callbacks.on_block_images_change)(new_value == "true"),
            "skill-commands" => {
                (callbacks.on_enable_skill_commands_change)(new_value == "true");
            }
            "steering-mode" => (callbacks.on_steering_mode_change)(match new_value {
                "all" => QueueMode::All,
                _ => QueueMode::OneAtATime,
            }),
            "follow-up-mode" => (callbacks.on_follow_up_mode_change)(match new_value {
                "all" => QueueMode::All,
                _ => QueueMode::OneAtATime,
            }),
            "transport" => (callbacks.on_transport_change)(match new_value {
                "sse" => Transport::Sse,
                "websocket" => Transport::Websocket,
                "websocket-cached" => Transport::WebsocketCached,
                _ => Transport::Auto,
            }),
            "http-idle-timeout" => {
                if let Some(choice) = HTTP_IDLE_TIMEOUT_CHOICES
                    .iter()
                    .find(|choice| choice.label == new_value)
                {
                    (callbacks.on_http_idle_timeout_ms_change)(choice.timeout_ms);
                }
            }
            "hide-thinking" => (callbacks.on_hide_thinking_block_change)(new_value == "true"),
            "mermaid-rendering" => {
                (callbacks.on_mermaid_rendering_mode_change)(match new_value {
                    "final" => MermaidRenderingMode::Final,
                    "streaming" => MermaidRenderingMode::Streaming,
                    _ => MermaidRenderingMode::Off,
                });
            }
            "cache-miss-notices" => {
                (callbacks.on_show_cache_miss_notices_change)(new_value == "true");
            }
            "collapse-changelog" => (callbacks.on_collapse_changelog_change)(new_value == "true"),
            "quiet-startup" => (callbacks.on_quiet_startup_change)(new_value == "true"),
            "install-telemetry" => {
                (callbacks.on_enable_install_telemetry_change)(new_value == "true");
            }
            "default-project-trust" => {
                if let Some(trust) = trust_from_label(new_value) {
                    (callbacks.on_default_project_trust_change)(trust);
                }
            }
            "double-escape-action" => (callbacks.on_double_escape_action_change)(new_value),
            "tree-filter-mode" => (callbacks.on_tree_filter_mode_change)(new_value),
            "show-hardware-cursor" => {
                (callbacks.on_show_hardware_cursor_change)(new_value == "true");
            }
            "editor-padding" => {
                (callbacks.on_editor_padding_x_change)(new_value.parse().unwrap_or_default());
            }
            "output-padding" => {
                (callbacks.on_output_pad_change)(u8::from(new_value != "0"));
            }
            "autocomplete-max-visible" => {
                (callbacks.on_autocomplete_max_visible_change)(
                    new_value.parse().unwrap_or_default(),
                );
            }
            "clear-on-shrink" => (callbacks.on_clear_on_shrink_change)(new_value == "true"),
            "terminal-progress" => {
                (callbacks.on_show_terminal_progress_change)(new_value == "true");
            }
            "tui-mode" => (callbacks.on_tui_mode_change)(match new_value {
                "fullscreen" => TuiMode::Fullscreen,
                _ => TuiMode::Regular,
            }),
            "fullscreen-exit-output" => {
                (callbacks.on_fullscreen_exit_output_change)(match new_value {
                    "resume-hint" => FullscreenExitOutput::ResumeHint,
                    _ => FullscreenExitOutput::Transcript,
                });
            }
            "fullscreen-scrollbar" => {
                (callbacks.on_fullscreen_scrollbar_change)(match new_value {
                    "always" => ScrollViewScrollbar::Always,
                    "hidden" => ScrollViewScrollbar::Hidden,
                    _ => ScrollViewScrollbar::Auto,
                });
            }
            "fullscreen-copy-on-select" => {
                (callbacks.on_fullscreen_copy_on_select_change)(new_value == "true");
            }
            "theme" => (callbacks.on_theme_change)(new_value),
            _ => {}
        });

        let settings_list = Rc::new(SettingsList::with_options(
            items,
            10,
            get_settings_list_theme(),
            on_change,
            on_cancel,
            SettingsListOptions {
                enable_search: true,
            },
        ));
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        let settings_component: Rc<dyn Component> = settings_list.clone();
        container.add_child(settings_component);
        container.add_child(Rc::new(DynamicBorder::themed()));
        Self {
            container,
            settings_list,
        }
    }

    /// The settings list, upstream's `getSettingsList`.
    #[must_use]
    pub fn get_settings_list(&self) -> Rc<SettingsList> {
        self.settings_list.clone()
    }
}

impl Component for SettingsSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        self.settings_list.handle_input(data);
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn invalidate(&self) {
        self.container.invalidate();
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }
}
