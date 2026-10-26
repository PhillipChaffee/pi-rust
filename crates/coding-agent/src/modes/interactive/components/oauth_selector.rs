//! OAuth provider selector at upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#204).
//!
//! The login/logout provider picker renders the options its host supplies; the
//! host owns provider projection, login, logout, and credential persistence.
//!
//! Restatements against upstream:
//!
//! - The optional `method: ApiKeyAuth | OAuthAuth` field is an enum because the
//!   structural union has no Rust equivalent. The selector reads only its
//!   display name; the host keeps the method for the login that follows.
//! - The env-var-list test is the same `[A-Z][A-Z0-9_]*(?:, [A-Z][A-Z0-9_]*)*`
//!   grammar hand-rolled, so the selector needs no regex dependency.
//! - `Input.onSubmit` is not bound: the confirm arm handles Enter before the
//!   input sees it, which is the only path upstream's submit handler serves.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pi_ai::auth::types::{ApiKeyAuth, AuthCheck, AuthType, OAuthAuth};
use pi_tui::components::input::Input;
use pi_tui::components::spacer::Spacer;
use pi_tui::components::truncated_text::TruncatedText;
use pi_tui::fuzzy::fuzzy_filter;
use pi_tui::keybindings::get_keybindings;
use pi_tui::keys::KeyParser;
use pi_tui::tui::{Component, Container, Focusable, TuiMouseEvent, TuiMouseEventResult};

use super::dynamic_border::DynamicBorder;
use crate::modes::interactive::theme::theme::{bold, theme};

/// Rows visible before scrolling, upstream's `maxVisible`.
const MAX_VISIBLE: usize = 8;

/// Which credential action the selector is choosing for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthSelectorMode {
    /// Choose a provider to configure.
    Login,
    /// Choose a stored credential to remove.
    Logout,
}

impl OAuthSelectorMode {
    const fn title(self) -> &'static str {
        match self {
            Self::Login => "Select provider to configure:",
            Self::Logout => "Select provider to logout:",
        }
    }
}

/// The optional auth method behind one provider option.
#[derive(Clone, Debug)]
pub enum AuthSelectorMethod {
    /// Api-key auth.
    ApiKey(ApiKeyAuth),
    /// OAuth auth.
    OAuth(OAuthAuth),
}

impl AuthSelectorMethod {
    /// The method's display name, upstream's `method.name`.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::ApiKey(auth) => &auth.name,
            Self::OAuth(auth) => &auth.name,
        }
    }
}

/// One selectable provider, upstream's `AuthSelectorProvider`.
#[derive(Clone, Debug)]
pub struct AuthSelectorProvider {
    /// Stable provider id, e.g. `anthropic`.
    pub id: String,
    /// Display name, e.g. `Anthropic`.
    pub name: String,
    /// Which credential type this option configures.
    pub auth_type: AuthType,
    /// The matching auth method from the provider registry.
    pub method: Option<AuthSelectorMethod>,
    /// Resolved availability; `None` renders as unconfigured.
    pub status: Option<AuthCheck>,
}

/// The auth-type label used in the picker, upstream's
/// `formatAuthSelectorProviderType`.
#[must_use]
pub const fn format_auth_selector_provider_type(auth_type: AuthType) -> &'static str {
    match auth_type {
        AuthType::OAuth => "subscription",
        AuthType::ApiKey => "API key",
    }
}

/// The same grammar as upstream's env-source regex, without a compiled regex.
fn is_env_var_list(source: &str) -> bool {
    source.split(", ").all(|name| {
        let mut chars = name.chars();
        chars.next().is_some_and(|first| first.is_ascii_uppercase())
            && chars.all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
    })
}

fn status_indicator(provider: &AuthSelectorProvider) -> String {
    let Some(status) = &provider.status else {
        return theme().fg("muted", " • unconfigured");
    };
    if status.auth_type != provider.auth_type {
        let label = match status.auth_type {
            AuthType::OAuth => "subscription configured",
            AuthType::ApiKey => "API key configured",
        };
        return theme().fg("muted", " • ") + &theme().fg("warning", label);
    }
    let Some(source) = status.source.as_deref().filter(|source| !source.is_empty()) else {
        return theme().fg("success", " ✓ configured");
    };
    if source == "OAuth" || source == "stored credential" {
        return theme().fg("success", " ✓ configured");
    }
    let source = if is_env_var_list(source) {
        format!("env: {source}")
    } else {
        source.to_string()
    };
    theme().fg("success", &format!(" ✓ {source}"))
}

/// A searchable provider list with a mode-specific title.
pub struct OAuthSelectorComponent {
    container: Container,
    list_container: Rc<Container>,
    search_input: Rc<Input>,
    all_providers: Vec<AuthSelectorProvider>,
    filtered_providers: RefCell<Vec<AuthSelectorProvider>>,
    selected_index: Cell<usize>,
    mode: OAuthSelectorMode,
    show_auth_type_labels: bool,
    on_select: Rc<dyn Fn(String, AuthType)>,
    on_cancel: Rc<dyn Fn()>,
    parser: KeyParser,
}

impl std::fmt::Debug for OAuthSelectorComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthSelectorComponent")
            .field("mode", &self.mode)
            .field("providers", &self.all_providers.len())
            .field("selected_index", &self.selected_index.get())
            .finish_non_exhaustive()
    }
}

impl OAuthSelectorComponent {
    /// Filters the title, provider list, and auth-type labels by mode.
    /// `on_select` receives the provider id and the credential type to handle.
    #[must_use]
    pub fn new(
        mode: OAuthSelectorMode,
        providers: Vec<AuthSelectorProvider>,
        on_select: Rc<dyn Fn(String, AuthType)>,
        on_cancel: Rc<dyn Fn()>,
        initial_search: Option<String>,
    ) -> Self {
        let show_auth_type_labels = providers
            .first()
            .is_some_and(|first| providers.iter().any(|p| p.auth_type != first.auth_type));
        let search_input = Rc::new(Input::new());
        if let Some(query) = initial_search.filter(|query| !query.is_empty()) {
            search_input.set_value(query);
        }
        let list_container = Rc::new(Container::default());
        let container = Container::default();
        container.add_child(Rc::new(DynamicBorder::themed()));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(TruncatedText::with_padding(
            theme().fg("accent", &bold(mode.title())),
            1,
            0,
        )));
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(search_input.clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(list_container.clone());
        container.add_child(Rc::new(Spacer::new()));
        container.add_child(Rc::new(DynamicBorder::themed()));
        let component = Self {
            container,
            list_container,
            search_input,
            all_providers: providers,
            filtered_providers: RefCell::new(Vec::new()),
            selected_index: Cell::new(0),
            mode,
            show_auth_type_labels,
            on_select,
            on_cancel,
            parser: KeyParser::new(),
        };
        let query = component.search_input.get_value();
        component.filter_providers(&query);
        component
    }

    fn filter_providers(&self, query: &str) {
        let filtered = if query.is_empty() {
            self.all_providers.clone()
        } else {
            fuzzy_filter(&self.all_providers, query, |provider| {
                format!(
                    "{} {} {} {}",
                    provider.name,
                    provider.id,
                    provider.auth_type,
                    provider
                        .method
                        .as_ref()
                        .map_or("", AuthSelectorMethod::name)
                )
            })
            .into_iter()
            .cloned()
            .collect()
        };
        let len = filtered.len();
        self.selected_index
            .set(self.selected_index.get().min(len.saturating_sub(1)));
        *self.filtered_providers.borrow_mut() = filtered;
        self.update_list();
    }

    fn update_list(&self) {
        self.list_container.clear();
        let providers = self.filtered_providers.borrow();
        let len = providers.len();
        let selected = self.selected_index.get();
        let start = selected
            .saturating_sub(MAX_VISIBLE / 2)
            .min(len.saturating_sub(MAX_VISIBLE));
        let end = (start + MAX_VISIBLE).min(len);
        for (index, provider) in providers.iter().enumerate().take(end).skip(start) {
            let is_selected = index == selected;
            let auth_label = if self.show_auth_type_labels {
                theme().fg(
                    "muted",
                    &format!(
                        " [{}]",
                        format_auth_selector_provider_type(provider.auth_type)
                    ),
                )
            } else {
                String::new()
            };
            let line = if is_selected {
                format!(
                    "{}{}{}{}",
                    theme().fg("accent", "→ "),
                    theme().fg("accent", &provider.name),
                    auth_label,
                    status_indicator(provider)
                )
            } else {
                format!(
                    "  {}{}{}",
                    theme().fg("text", &provider.name),
                    auth_label,
                    status_indicator(provider)
                )
            };
            self.add_line(line);
        }
        if start > 0 || end < len {
            self.add_line(theme().fg("muted", &format!("  ({}/{len})", selected + 1)));
        }
        if len == 0 {
            let message = if self.all_providers.is_empty() {
                match self.mode {
                    OAuthSelectorMode::Login => "No providers available",
                    OAuthSelectorMode::Logout => "No providers logged in. Use /login first.",
                }
            } else {
                "No matching providers"
            };
            self.add_line(theme().fg("muted", &format!("  {message}")));
        }
    }

    fn add_line(&self, line: String) {
        self.list_container
            .add_child(Rc::new(TruncatedText::with_padding(line, 1, 0)));
    }

    fn confirm(&self) {
        let selected = self
            .filtered_providers
            .borrow()
            .get(self.selected_index.get())
            .cloned();
        if let Some(provider) = selected {
            (self.on_select)(provider.id, provider.auth_type);
        }
    }
}

impl Component for OAuthSelectorComponent {
    fn render(&self, width: usize) -> Vec<String> {
        self.container.render(width)
    }

    fn handle_input(&self, data: &str) {
        let matches = |action| get_keybindings().matches(&self.parser, data, action);
        if matches("tui.select.up") {
            if self.filtered_providers.borrow().is_empty() {
                return;
            }
            self.selected_index
                .set(self.selected_index.get().saturating_sub(1));
            self.update_list();
        } else if matches("tui.select.down") {
            let len = self.filtered_providers.borrow().len();
            if len == 0 {
                return;
            }
            self.selected_index
                .set(self.selected_index.get().saturating_add(1).min(len - 1));
            self.update_list();
        } else if matches("tui.select.confirm") {
            self.confirm();
        } else if matches("tui.select.cancel") {
            (self.on_cancel)();
        } else {
            self.search_input.handle_input(data);
            let query = self.search_input.get_value();
            self.filter_providers(&query);
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

    fn as_focusable(&self) -> Option<&dyn Focusable> {
        Some(self)
    }

    fn handle_mouse(&self, event: &TuiMouseEvent) -> Option<TuiMouseEventResult> {
        self.container.handle_mouse(event)
    }
}

impl Focusable for OAuthSelectorComponent {
    fn set_focused(&self, focused: bool) {
        self.search_input.set_focused(focused);
    }

    fn is_focused(&self) -> bool {
        self.search_input.is_focused()
    }
}
