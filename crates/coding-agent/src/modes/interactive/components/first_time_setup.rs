//! The first-run setup dialog, upstream's
//! `src/modes/interactive/components/first-time-setup.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#131).
//!
//! Two steps — the theme choice, then the anonymous-usage opt-in — rebuilt on
//! every keystroke so a previewed theme recolors the whole dialog. The dialog
//! owns no persistence: [`FirstTimeSetupResult`] goes back to the caller, and
//! `cli::startup_ui` writes it to settings.
//!
//! Restatements: the callbacks are `Rc<dyn Fn>` closures, the synchronous
//! port of upstream's function properties.

use std::cell::Cell;
use std::rc::Rc;

use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, raw_key_hint};
use crate::config::APP_NAME;
use crate::modes::interactive::theme::theme::{TerminalTheme, bold, theme};

/// The finished decision, upstream's `FirstTimeSetupResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstTimeSetupResult {
    /// The chosen theme, upstream's `theme`.
    pub theme: TerminalTheme,
    /// Whether anonymous usage data may be shared, upstream's
    /// `shareAnalytics`.
    pub share_analytics: bool,
}

/// Callbacks and the detected appearance, upstream's
/// `FirstTimeSetupOptions`.
pub struct FirstTimeSetupOptions {
    /// The appearance the terminal probe read; selects the initial row.
    pub detected_theme: TerminalTheme,
    /// Applies a preview while the theme step is focused, upstream's
    /// `onThemePreview`.
    pub on_theme_preview: Rc<dyn Fn(TerminalTheme)>,
    /// Receives the finished decision, upstream's `onSubmit`.
    pub on_submit: Rc<dyn Fn(FirstTimeSetupResult)>,
    /// Dismissal without a decision, upstream's `onCancel`.
    pub on_cancel: Rc<dyn Fn()>,
}

impl std::fmt::Debug for FirstTimeSetupOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FirstTimeSetupOptions")
            .field("detected_theme", &self.detected_theme)
            .finish_non_exhaustive()
    }
}

/// The theme step's rows, upstream's `THEME_OPTIONS`.
const THEME_OPTIONS: [(TerminalTheme, &str); 2] = [
    (TerminalTheme::Dark, "Dark"),
    (TerminalTheme::Light, "Light"),
];

/// The analytics step's rows, upstream's `ANALYTICS_OPTIONS`.
const ANALYTICS_OPTIONS: [(bool, &str); 2] =
    [(true, "Share anonymous usage data"), (false, "Don't share")];

/// The block-letter logo, upstream's `SETUP_LOGO_LINES`.
const SETUP_LOGO_LINES: [&str; 4] = ["██████", "██  ██", "████  ██", "██    ██"];

/// The analytics explanation, upstream's wrapped paragraph.
const ANALYTICS_DETAIL: &str = "Opting in stores a tracking identifier in settings.json and enables anonymous\nusage analytics. This helps us to better debug, reproduce, and resolve issues\nand bugs within Pi. You can observe what is shared using /privacy and make\nchanges anytime in settings.json.";

/// Which step the dialog shows, upstream's `"theme" | "analytics"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SetupStep {
    Theme,
    Analytics,
}

/// The first-run dialog, upstream's `FirstTimeSetupComponent`.
pub struct FirstTimeSetupComponent {
    container: Container,
    options: FirstTimeSetupOptions,
    step: Cell<SetupStep>,
    theme_index: Cell<usize>,
    analytics_index: Cell<usize>,
    parser: KeyParser,
}

impl std::fmt::Debug for FirstTimeSetupComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FirstTimeSetupComponent")
            .field("step", &self.step.get())
            .field("theme_index", &self.theme_index.get())
            .field("analytics_index", &self.analytics_index.get())
            .finish_non_exhaustive()
    }
}

impl FirstTimeSetupComponent {
    /// Build the dialog, starting on the detected theme, upstream's
    /// constructor body.
    #[must_use]
    pub fn new(options: FirstTimeSetupOptions) -> Self {
        let theme_index = THEME_OPTIONS
            .iter()
            .position(|(value, _)| *value == options.detected_theme)
            .unwrap_or(0);
        let component = Self {
            container: Container::default(),
            options,
            step: Cell::new(SetupStep::Theme),
            theme_index: Cell::new(theme_index),
            analytics_index: Cell::new(0),
            parser: KeyParser::new(),
        };
        component.update();
        component
    }

    /// Rebuild the whole dialog, upstream's `update`: every child is new so a
    /// theme preview recolors all text.
    fn update(&self) {
        self.container.clear();
        self.container.add_child(Rc::new(DynamicBorder::themed()));
        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &SETUP_LOGO_LINES.join("\n")),
            1,
            0,
        )));
        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "accent",
                &bold(&format!("Welcome to {APP_NAME}, the minimal coding agent.")),
            ),
            1,
            0,
        )));
        self.container.add_child(Rc::new(Spacer::new()));

        match self.step.get() {
            SetupStep::Theme => {
                self.container.add_child(Rc::new(Text::with_padding(
                    theme().fg("text", "Pick a theme."),
                    1,
                    0,
                )));
                self.container.add_child(Rc::new(Text::with_padding(
                    theme().fg(
                        "muted",
                        &format!(
                            "Detected system appearance: {}",
                            self.options.detected_theme.theme_name()
                        ),
                    ),
                    1,
                    0,
                )));
                self.container.add_child(Rc::new(Spacer::new()));
                self.add_option_list(&THEME_OPTIONS, self.theme_index.get());
            }
            SetupStep::Analytics => {
                self.container.add_child(Rc::new(Text::with_padding(
                    theme().fg("text", "Opt-in to anonymous usage data sharing?"),
                    1,
                    0,
                )));
                self.container.add_child(Rc::new(Text::with_padding(
                    theme().fg("muted", ANALYTICS_DETAIL),
                    1,
                    0,
                )));
                self.container.add_child(Rc::new(Spacer::new()));
                self.add_option_list(&ANALYTICS_OPTIONS, self.analytics_index.get());
            }
        }

        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(Text::with_padding(
            raw_key_hint("↑↓", "navigate")
                + "  "
                + &key_hint(
                    "tui.select.confirm",
                    match self.step.get() {
                        SetupStep::Theme => "continue",
                        SetupStep::Analytics => "finish",
                    },
                )
                + "  "
                + &key_hint("tui.select.cancel", "skip setup"),
            1,
            0,
        )));
        self.container.add_child(Rc::new(Spacer::new()));
        self.container.add_child(Rc::new(DynamicBorder::themed()));
    }

    /// Render one option list, upstream's `addOptionList`.
    fn add_option_list(&self, options: &[(impl Copy, &str)], selected_index: usize) {
        for (index, (_, label)) in options.iter().enumerate() {
            let is_selected = index == selected_index;
            let prefix = if is_selected {
                theme().fg("accent", "→ ")
            } else {
                "  ".to_string()
            };
            let text = if is_selected {
                theme().fg("accent", label)
            } else {
                theme().fg("text", label)
            };
            self.container
                .add_child(Rc::new(Text::with_padding(prefix + &text, 1, 0)));
        }
    }

    /// Move the focused step's selection, clamped at both ends, upstream's
    /// `moveSelection`.
    fn move_selection(&self, delta: isize) {
        let clamp = |index: usize, last: usize| index.saturating_add_signed(delta).min(last);
        match self.step.get() {
            SetupStep::Theme => {
                let next = clamp(self.theme_index.get(), THEME_OPTIONS.len() - 1);
                if next != self.theme_index.get() {
                    self.theme_index.set(next);
                    (self.options.on_theme_preview)(THEME_OPTIONS[next].0);
                }
            }
            SetupStep::Analytics => {
                self.analytics_index.set(clamp(
                    self.analytics_index.get(),
                    ANALYTICS_OPTIONS.len() - 1,
                ));
            }
        }
        self.update();
    }

    /// The keyboard dispatch, upstream `handleInput`.
    pub fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.up") || data == "k" {
            self.move_selection(-1);
        } else if matches("tui.select.down") || data == "j" {
            self.move_selection(1);
        } else if matches("tui.select.confirm") || data == "\n" {
            match self.step.get() {
                SetupStep::Theme => {
                    self.step.set(SetupStep::Analytics);
                    self.update();
                }
                SetupStep::Analytics => {
                    (self.options.on_submit)(FirstTimeSetupResult {
                        theme: THEME_OPTIONS[self.theme_index.get()].0,
                        share_analytics: ANALYTICS_OPTIONS[self.analytics_index.get()].0,
                    });
                }
            }
        } else if matches("tui.select.cancel") {
            (self.options.on_cancel)();
        }
    }
}

impl Component for FirstTimeSetupComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn wants_input(&self) -> bool {
        true
    }

    fn handle_input(&self, data: &str) {
        Self::handle_input(self, data);
    }

    fn children(&self) -> Vec<Rc<dyn Component>> {
        self.container.children()
    }
}
