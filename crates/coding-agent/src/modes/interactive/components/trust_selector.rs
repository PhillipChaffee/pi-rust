//! Project-trust picker at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//! Selection returns the option builder's updates; only the caller persists them.

use std::cell::Cell;
use std::rc::Rc;

use pi_tui::components::spacer::Spacer;
use pi_tui::components::text::Text;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use super::keybinding_hints::{key_hint, raw_key_hint};
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::trust_manager::{
    ProjectTrustOption, ProjectTrustStoreEntry, ProjectTrustUpdate, get_project_trust_options,
};

/// The option's decision and exact store updates, without its display fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustSelection {
    /// Whether the selected option trusts the project.
    pub trusted: bool,
    /// Includes the child-path deletion when trusting a parent.
    pub updates: Vec<ProjectTrustUpdate>,
}

/// The saved decision and the current session decision can differ.
pub struct TrustSelectorOptions {
    /// Working directory displayed verbatim; options canonicalize it through the trust manager.
    pub cwd: String,
    /// Nearest persisted decision, including ancestor decisions; `None` displays `none`.
    pub saved_decision: Option<ProjectTrustStoreEntry>,
    /// Effective trust for this session, independent of the saved marker.
    pub project_trusted: bool,
    /// Receives updates for the caller to persist.
    pub on_select: Rc<dyn Fn(TrustSelection)>,
    /// Dismissal makes no trust changes.
    pub on_cancel: Rc<dyn Fn()>,
}

impl std::fmt::Debug for TrustSelectorOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustSelectorOptions")
            .field("cwd", &self.cwd)
            .field("saved_decision", &self.saved_decision)
            .field("project_trusted", &self.project_trusted)
            .finish_non_exhaustive()
    }
}

/// A clamped, keyboard-navigable list of persistent project-trust choices.
pub struct TrustSelectorComponent {
    container: Container,
    list_container: Rc<Container>,
    selected_index: Cell<usize>,
    trust_options: Vec<ProjectTrustOption>,
    saved_decision: Option<ProjectTrustStoreEntry>,
    on_select: Rc<dyn Fn(TrustSelection)>,
    on_cancel: Rc<dyn Fn()>,
    parser: KeyParser,
}

impl std::fmt::Debug for TrustSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrustSelectorComponent")
            .field("selected_index", &self.selected_index.get())
            .field("trust_options", &self.trust_options)
            .finish_non_exhaustive()
    }
}

impl TrustSelectorComponent {
    /// Preselects an exact saved option; inherited decisions without an option start at Trust.
    #[must_use]
    pub fn new(options: TrustSelectorOptions) -> Self {
        let trust_options = get_project_trust_options(&options.cwd, false);
        let selected_index = trust_options
            .iter()
            .position(|option| is_saved_option(option, options.saved_decision.as_ref()))
            .unwrap_or_default();
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("accent", &bold("Project trust")),
            1,
            0,
        )));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("muted", &options.cwd),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        let decision = options.saved_decision.as_ref().map_or_else(
            || "none".to_string(),
            |decision| {
                let label = if decision.decision {
                    "trusted"
                } else {
                    "untrusted"
                };
                let inherited = trust_options
                    .first()
                    .and_then(|option| option.saved_path.as_ref())
                    .is_some_and(|path| path != &decision.path);
                format!(
                    "{label} ({}{})",
                    if inherited { "inherited from " } else { "" },
                    decision.path
                )
            },
        );
        container.add_child(Rc::new(Text::with_padding(
            theme().fg("muted", &format!("Saved decision: {decision}")),
            1,
            0,
        )));
        container.add_child(Rc::new(Text::with_padding(
            theme().fg(
                "muted",
                &format!(
                    "Current session: {}",
                    if options.project_trusted {
                        "trusted"
                    } else {
                        "untrusted"
                    }
                ),
            ),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        let list_container = Rc::new(Container::default());
        container.add_child(list_container.clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(Text::with_padding(
            raw_key_hint("↑↓", "navigate")
                + "  "
                + &key_hint("tui.select.confirm", "save")
                + "  "
                + &key_hint("tui.select.cancel", "cancel"),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));
        let selector = Self {
            container,
            list_container,
            selected_index: Cell::new(selected_index),
            trust_options,
            saved_decision: options.saved_decision,
            on_select: options.on_select,
            on_cancel: options.on_cancel,
            parser: KeyParser::new(),
        };
        selector.update_list();
        selector
    }

    fn update_list(&self) {
        self.list_container.clear();
        for (index, option) in self.trust_options.iter().enumerate() {
            let selected = index == self.selected_index.get();
            let current = if is_saved_option(option, self.saved_decision.as_ref()) {
                theme().fg("accent", "✓ ")
            } else {
                "  ".to_string()
            };
            let prefix = if selected {
                theme().fg("accent", "→ ")
            } else {
                "  ".to_string()
            };
            let label = theme().fg(if selected { "accent" } else { "text" }, &option.label);
            self.list_container.add_child(Rc::new(Text::with_padding(
                format!("{prefix}{current}{label}"),
                1,
                0,
            )));
        }
    }
}

fn is_saved_option(option: &ProjectTrustOption, saved: Option<&ProjectTrustStoreEntry>) -> bool {
    saved.is_some_and(|saved| {
        option.saved_path.as_deref() == Some(saved.path.as_str())
            && saved.decision == option.trusted
    })
}

impl Component for TrustSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.up") || data == "k" {
            self.selected_index
                .set(self.selected_index.get().saturating_sub(1));
            self.update_list();
        } else if matches("tui.select.down") || data == "j" {
            self.selected_index.set(
                (self.selected_index.get() + 1).min(self.trust_options.len().saturating_sub(1)),
            );
            self.update_list();
        } else if matches("tui.select.confirm") || data == "\n" {
            if let Some(selected) = self.trust_options.get(self.selected_index.get()) {
                (self.on_select)(TrustSelection {
                    trusted: selected.trusted,
                    updates: selected.updates.clone(),
                });
            }
        } else if matches("tui.select.cancel") {
            (self.on_cancel)();
        }
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
