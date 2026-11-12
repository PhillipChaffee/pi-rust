//! Selectors, login, and the slash-command handlers.

use std::rc::Rc;
use std::sync::Arc;

use pi_agent_core::types::ThinkingLevel;
use pi_ai::auth::types::AuthType;
use pi_ai::types::Model;
use pi_tui::autocomplete::{
    AutocompleteProvider, CombinedAutocompleteProvider, CommandEntry, SlashCommand,
};
use pi_tui::tui::{Component as TuiComponent, TuiMode};

use crate::core::agent_session::{ModelMutationOptions, NavigateTreeOptions};
use crate::core::agent_session_runtime::{ForkOptions, NewSessionRequest, SwitchSessionOptions};
use crate::modes::interactive::theme::theme::{bold, theme};
use crate::settings_manager::SettingsStorageKind;
use pi_tui::components::{Markdown, Spacer, Text};

use super::{FrameCommand, InteractiveMode};

/// One created selector: its component, the focus target, and an optional
/// disposer.
pub(crate) struct SelectorParts {
    component: Rc<dyn TuiComponent>,
    focus: Rc<dyn TuiComponent>,
    dispose: Option<Rc<dyn Fn()>>,
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show a selector in place of the editor, upstream's `showSelector`.
    pub(crate) fn show_selector(&self, create: impl FnOnce(Rc<dyn Fn()>) -> SelectorParts) {
        let container = Rc::clone(&self.editor_container);
        let tui = Rc::clone(&self.tui);
        let editor: Rc<dyn TuiComponent> = self.editor();
        let done: Rc<dyn Fn()> = Rc::new(move || {
            container.clear();
            container.add_child(Rc::clone(&editor));
            tui.set_focus(Some(Rc::clone(&editor)));
        });
        let parts = create(Rc::clone(&done));
        self.dispose_active_selector();
        self.active_selector_token.borrow_mut().replace(1);
        *self.active_selector_dispose.borrow_mut() = parts.dispose;
        self.editor_container.clear();
        self.editor_container.add_child(Rc::clone(&parts.component));
        self.tui.set_focus(Some(Rc::clone(&parts.focus)));
        self.tui.request_render(false);
    }

    /// Dispose the active selector and restore the editor.
    pub(crate) fn dispose_active_selector(&self) {
        self.active_selector_token.borrow_mut().take();
        if let Some(dispose) = self.active_selector_dispose.borrow_mut().take() {
            dispose();
        }
        if self.extension_selector.borrow().is_none()
            && self.extension_input.borrow().is_none()
            && self.extension_editor.borrow().is_none()
        {
            let editor: Rc<dyn TuiComponent> = self.editor();
            self.editor_container.clear();
            self.editor_container.add_child(Rc::clone(&editor));
            self.tui.set_focus(Some(editor));
        }
    }

    /// Build the combined autocomplete provider, upstream's
    /// `createBaseAutocompleteProvider`.
    pub fn create_base_autocomplete_provider(&self) -> Arc<dyn AutocompleteProvider> {
        let models: Vec<Model> = {
            let scoped = self.session().scoped_models();
            if scoped.is_empty() {
                self.session().model_runtime().get_available_snapshot()
            } else {
                scoped.into_iter().map(|scoped| scoped.model).collect()
            }
        };
        let mut model_completions: Option<pi_tui::autocomplete::SlashArgumentCompletionsFn> =
            Some(Box::new(move |prefix: &str| {
                fuzzy_autocomplete_items(
                    &models,
                    prefix,
                    |model| {
                        crate::modes::interactive::model_search::get_model_search_text(
                            &model.id,
                            &model.provider.0,
                            Some(&model.name),
                        )
                    },
                    |model| pi_tui::autocomplete::AutocompleteItem {
                        value: format!("{}/{}", model.provider.0, model.id),
                        label: model.id.clone(),
                        description: Some(model.provider.0.clone()),
                    },
                )
            }));
        let thinking_levels = self.session().available_thinking_levels();
        let mut thinking_completions: Option<pi_tui::autocomplete::SlashArgumentCompletionsFn> =
            Some(Box::new(move |prefix: &str| {
                fuzzy_autocomplete_items(
                    &thinking_levels,
                    prefix,
                    |level| thinking_level_name(*level).to_owned(),
                    |level| {
                        let name = thinking_level_name(*level).to_owned();
                        pi_tui::autocomplete::AutocompleteItem {
                            value: name.clone(),
                            label: name,
                            description: None,
                        }
                    },
                )
            }));
        let login_providers = login_completion_options(&self.get_login_provider_options(None));
        let mut login_completions: Option<pi_tui::autocomplete::SlashArgumentCompletionsFn> =
            Some(Box::new(move |prefix: &str| {
                fuzzy_autocomplete_items(
                    &login_providers,
                    prefix,
                    |provider| provider.search_text(),
                    |provider| pi_tui::autocomplete::AutocompleteItem {
                        value: provider.id.clone(),
                        label: provider.id.clone(),
                        description: Some(provider.description()),
                    },
                )
            }));
        let mut commands: Vec<CommandEntry> = crate::core::slash_commands::BUILTIN_SLASH_COMMANDS
            .iter()
            .map(|command| {
                let get_argument_completions = match command.name {
                    "model" => model_completions.take(),
                    "thinking" => thinking_completions.take(),
                    "login" => login_completions.take(),
                    _ => None,
                };
                CommandEntry::Slash(SlashCommand {
                    name: command.name.to_owned(),
                    description: Some(command.description.to_owned()),
                    argument_hint: command.argument_hint.map(str::to_owned),
                    get_argument_completions,
                })
            })
            .collect();
        let templates = self.session().prompt_templates();
        commands.extend(templates.into_iter().map(|template| {
            CommandEntry::Slash(SlashCommand {
                name: template.name,
                description: Some(template.description),
                argument_hint: template.argument_hint,
                get_argument_completions: None,
            })
        }));
        let builtin_names: std::collections::BTreeSet<String> = commands
            .iter()
            .filter_map(|entry| match entry {
                CommandEntry::Slash(command) => Some(command.name.clone()),
                CommandEntry::Item(item) => Some(item.value.clone()),
            })
            .collect();
        let extension_commands = self
            .session()
            .extension_runner()
            .get_registered_commands()
            .into_iter()
            .filter(|command| !builtin_names.contains(&command.name))
            .map(|command| {
                CommandEntry::Slash(SlashCommand {
                    name: command.invocation_name,
                    description: command.description,
                    argument_hint: None,
                    get_argument_completions: None,
                })
            });
        commands.extend(extension_commands);
        let mut skill_commands = std::collections::HashMap::new();
        if self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_enable_skill_commands()
        {
            for skill in self
                .session()
                .with_resource_loader(|loader| loader.get_skills().skills)
            {
                let name = format!("skill:{}", skill.name);
                skill_commands.insert(name.clone(), skill.file_path);
                commands.push(CommandEntry::Slash(SlashCommand {
                    name,
                    description: Some(skill.description),
                    argument_hint: None,
                    get_argument_completions: None,
                }));
            }
        }
        *self.skill_commands.borrow_mut() = skill_commands;
        let cwd = crate::core::agent_session::lock(&self.session().session_manager)
            .cwd()
            .to_owned();
        Arc::new(CombinedAutocompleteProvider::new(
            commands,
            cwd,
            self.fd_path.borrow().clone(),
        ))
    }

    /// Build and install the autocomplete provider, upstream's
    /// `setupAutocompleteProvider`.
    pub fn setup_autocomplete_provider(&self) {
        let mut provider = self.create_base_autocomplete_provider();
        let wrappers: Vec<super::AutocompleteProviderFactory> =
            self.autocomplete_provider_wrappers.borrow().clone();
        let mut trigger_characters: Vec<String> = Vec::new();
        for wrapper in wrappers {
            provider = wrapper(provider);
            trigger_characters.extend(provider.trigger_characters());
        }
        if !trigger_characters.is_empty() {
            let mut merged: Vec<String> = Vec::new();
            for character in trigger_characters {
                if !merged.contains(&character) {
                    merged.push(character);
                }
            }
            provider = Arc::new(TriggerCharacterProvider {
                inner: provider,
                trigger_characters: merged,
            });
        }
        self.default_editor
            .set_autocomplete_provider(Arc::clone(&provider));
        if self.editor_component_factory.borrow().is_some() {
            self.editor()
                .set_autocomplete_provider(Arc::clone(&provider));
        }
        *self.autocomplete_provider.borrow_mut() = Some(provider);
    }

    /// Register one in-process autocomplete wrapper and rebuild, upstream's
    /// `addAutocompleteProvider` on the extension UI context.
    pub fn add_autocomplete_provider(&self, wrapper: super::AutocompleteProviderFactory) {
        self.autocomplete_provider_wrappers
            .borrow_mut()
            .push(wrapper);
        self.setup_autocomplete_provider();
    }

    /// The provider installed on the editor, upstream's `autocompleteProvider`.
    #[must_use]
    pub fn autocomplete_provider(&self) -> Option<Arc<dyn AutocompleteProvider>> {
        self.autocomplete_provider.borrow().clone()
    }

    /// Cycle the thinking level, upstream's `cycleThinkingLevel`.
    pub(crate) fn cycle_thinking_level(&self) {
        match self
            .session()
            .cycle_thinking_level(ModelMutationOptions::default())
        {
            Some(level) => {
                self.footer.invalidate();
                self.update_editor_border_color();
                self.show_status(&format!("Thinking level: {}", thinking_level_name(level)));
            }
            None => self.show_status("Current model does not support thinking"),
        }
    }

    /// Cycle the model, upstream's `cycleModel`.
    pub(crate) async fn cycle_model(&self, direction: crate::core::agent_session::CycleDirection) {
        match self
            .session()
            .cycle_model(direction, ModelMutationOptions::default())
        {
            Some(result) => {
                self.footer.invalidate();
                self.update_editor_border_color();
                let thinking =
                    if result.model.reasoning && result.thinking_level != ThinkingLevel::Off {
                        format!(
                            " (thinking: {})",
                            thinking_level_name(result.thinking_level)
                        )
                    } else {
                        String::new()
                    };
                self.show_status(&format!(
                    "Switched to {}{thinking}",
                    if result.model.name.is_empty() {
                        result.model.id.clone()
                    } else {
                        result.model.name.clone()
                    }
                ));
                self.maybe_warn_about_anthropic_subscription_auth(Some(&result.model))
                    .await;
            }
            None => {
                let message = if self.session().scoped_models().is_empty() {
                    "Only one model available"
                } else {
                    "Only one model in scope"
                };
                self.show_status(message);
            }
        }
    }

    /// Set the session model.
    pub(crate) async fn set_model(&self, model: Model, persist: bool) {
        match self
            .session()
            .set_model(model.clone(), ModelMutationOptions { persist })
            .await
        {
            Ok(()) => {
                self.update_available_provider_count().await;
                self.footer.invalidate();
                self.update_editor_border_color();
                let message = if persist {
                    format!("Default model: {}/{}", model.provider.0, model.id)
                } else {
                    format!("Model: {}", model.id)
                };
                self.show_status(&message);
                self.maybe_warn_about_anthropic_subscription_auth(Some(&model))
                    .await;
                self.check_daxnuts_easter_egg(&model);
            }
            Err(error) => self.show_error(&error),
        }
    }
}

/// One deduplicated login provider completion row, upstream's
/// `LoginProviderCompletionOption`.
#[derive(Clone, Debug)]
struct LoginCompletionOption {
    id: String,
    name: String,
    auth_types: Vec<AuthType>,
}

impl LoginCompletionOption {
    /// The fuzzy-search text, upstream's `getLoginProviderSearchText`.
    fn search_text(&self) -> String {
        let auth_types = self
            .auth_types
            .iter()
            .map(|auth_type| {
                format!(
                    "{} {}",
                    match auth_type {
                        AuthType::OAuth => "oauth",
                        AuthType::ApiKey => "api_key",
                    },
                    crate::modes::interactive::components::oauth_selector::format_auth_selector_provider_type(*auth_type)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {} {auth_types}", self.id, self.name)
    }

    /// The picker description, upstream's
    /// `formatLoginProviderCompletionDescription`.
    fn description(&self) -> String {
        let auth_types = self
            .auth_types
            .iter()
            .map(|auth_type| {
                crate::modes::interactive::components::oauth_selector::format_auth_selector_provider_type(*auth_type)
            })
            .collect::<Vec<_>>()
            .join("/");
        if self.name == self.id {
            auth_types
        } else {
            format!("{} · {auth_types}", self.name)
        }
    }
}

/// Deduplicate provider options by id, merging auth types, upstream's
/// `getLoginProviderCompletionOptions`.
fn login_completion_options(
    providers: &[crate::modes::interactive::components::oauth_selector::AuthSelectorProvider],
) -> Vec<LoginCompletionOption> {
    let mut by_id: Vec<LoginCompletionOption> = Vec::new();
    for provider in providers {
        if let Some(existing) = by_id.iter_mut().find(|option| option.id == provider.id) {
            if !existing.auth_types.contains(&provider.auth_type) {
                existing.auth_types.push(provider.auth_type);
                existing
                    .auth_types
                    .sort_by_key(|auth_type| match auth_type {
                        AuthType::OAuth => 0,
                        AuthType::ApiKey => 1,
                    });
            }
            continue;
        }
        by_id.push(LoginCompletionOption {
            id: provider.id.clone(),
            name: provider.name.clone(),
            auth_types: vec![provider.auth_type],
        });
    }
    by_id.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.name.cmp(&right.name))
    });
    by_id
}

/// Fuzzy-filter `items` and map them, upstream's
/// `createFuzzyAutocompleteItems`; `None` when nothing matches.
fn fuzzy_autocomplete_items<T>(
    items: &[T],
    prefix: &str,
    get_text: impl Fn(&T) -> String,
    to_item: impl Fn(&T) -> pi_tui::autocomplete::AutocompleteItem,
) -> Option<Vec<pi_tui::autocomplete::AutocompleteItem>> {
    let filtered = pi_tui::fuzzy::fuzzy_filter(items, prefix, get_text);
    if filtered.is_empty() {
        return None;
    }
    Some(filtered.into_iter().map(to_item).collect())
}

/// A provider whose trigger characters are the merged set of every wrapper
/// upstream sets by assignment on the final provider.
struct TriggerCharacterProvider {
    inner: Arc<dyn AutocompleteProvider>,
    trigger_characters: Vec<String>,
}

impl AutocompleteProvider for TriggerCharacterProvider {
    fn trigger_characters(&self) -> Vec<String> {
        self.trigger_characters.clone()
    }

    fn get_suggestions(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        options: &pi_tui::autocomplete::AutocompleteQueryOptions,
    ) -> Option<pi_tui::autocomplete::AutocompleteSuggestions> {
        self.inner
            .get_suggestions(lines, cursor_line, cursor_col, options)
    }

    fn apply_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
        item: &pi_tui::autocomplete::AutocompleteItem,
        prefix: &str,
    ) -> pi_tui::autocomplete::AutocompleteApplyResult {
        self.inner
            .apply_completion(lines, cursor_line, cursor_col, item, prefix)
    }

    fn should_trigger_file_completion(
        &self,
        lines: &[String],
        cursor_line: usize,
        cursor_col: usize,
    ) -> bool {
        self.inner
            .should_trigger_file_completion(lines, cursor_line, cursor_col)
    }
}

/// The wire name of a thinking level, upstream's string union.
fn thinking_level_name(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::Xhigh => "xhigh",
        ThinkingLevel::Max => "max",
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Handle `/thinking [term]`, upstream's `handleThinkingCommand`.
    pub(crate) fn handle_thinking_command(&self, search_term: Option<&str>) {
        let available = self.session().available_thinking_levels();
        let Some(search_term) = search_term.filter(|term| !term.trim().is_empty()) else {
            self.show_thinking_selector();
            return;
        };
        let normalized = search_term.trim().to_lowercase();
        let level = available
            .iter()
            .find(|level| thinking_level_name(**level) == normalized);
        match level {
            Some(level) => self.select_thinking_level(*level, false),
            None => {
                let names = available
                    .iter()
                    .map(|level| thinking_level_name(*level))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.show_error(&format!(
                    "Unknown thinking level \"{search_term}\". Available levels: {names}."
                ));
            }
        }
    }

    /// Apply a thinking level.
    pub(crate) fn select_thinking_level(&self, level: ThinkingLevel, persist: bool) {
        self.session()
            .set_thinking_level(level, ModelMutationOptions { persist });
        self.footer.invalidate();
        self.update_editor_border_color();
        self.show_status(&format!(
            "{}: {}",
            if persist {
                "Default thinking level"
            } else {
                "Thinking level"
            },
            thinking_level_name(level)
        ));
    }

    /// Show the thinking-level selector.
    pub(crate) fn show_thinking_selector(&self) {
        self.show_selector(|done| {
            let select_done = Rc::clone(&done);
            let cancel_done = Rc::clone(&done);
            let commands = self.commands();
            let default_commands = commands.clone();
            let current = self.session().thinking_level();
            let available = self.session().available_thinking_levels();
            let default = self
                .settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_default_thinking_level();
            let selector = Rc::new(
                crate::modes::interactive::components::thinking_selector::ThinkingSelectorComponent::new(
                    current,
                    &available,
                    Rc::new(move |level| {
                        let _ = commands.send(FrameCommand::SelectThinkingLevel {
                            level,
                            persist: false,
                        });
                        select_done();
                    }),
                    Rc::new(move || cancel_done()),
                    Some(Rc::new(move |level| {
                        let _ = default_commands.send(FrameCommand::SelectThinkingLevel {
                            level,
                            persist: true,
                        });
                    })),
                    default,
                ),
            );
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }

    /// Handle `/model [term]`, upstream's `handleModelCommand`.
    pub(crate) async fn handle_model_command(&self, search_term: Option<String>) {
        let Some(search_term) = search_term.filter(|term| !term.trim().is_empty()) else {
            self.show_model_selector(None);
            return;
        };
        if let Some(model) = self.find_exact_model_match(&search_term).await {
            self.set_model(model, false).await;
            return;
        }
        self.show_model_selector(Some(&search_term));
    }

    /// Find an exact model reference, refreshing the catalogs when needed.
    pub(crate) async fn find_exact_model_match(&self, search_term: &str) -> Option<Model> {
        let scoped = self.session().scoped_models();
        let cached: Vec<Model> = if scoped.is_empty() {
            self.session().model_runtime().get_available_snapshot()
        } else {
            scoped.into_iter().map(|scoped| scoped.model).collect()
        };
        let matched = crate::model_resolver::find_exact_model_reference_match(search_term, &cached);
        if matched.is_some() || !self.session().scoped_models().is_empty() {
            return matched;
        }
        self.show_status("Refreshing model catalogs…");
        let runtime = self.session().model_runtime().clone();
        let signal = tokio_util::sync::CancellationToken::new();
        let _ = crate::modes::interactive::model_catalog_refresh::refresh_model_catalogs(
            &runtime, &signal,
        )
        .await;
        crate::model_resolver::find_exact_model_reference_match(
            search_term,
            &self.session().model_runtime().get_available_snapshot(),
        )
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show the model selector.
    pub(crate) fn show_model_selector(&self, initial_search: Option<&str>) {
        self.show_selector(|done| {
            let commands = self.commands();
            let cancel_done = Rc::clone(&done);
            let commands_default = commands.clone();
            let settings = self.settings();
            let default_provider = settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_default_provider();
            let default_model_id = settings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_default_model();
            let default_model =
                default_provider
                    .zip(default_model_id)
                    .and_then(|(provider, id)| {
                        self.session()
                            .model_runtime()
                            .get_available_snapshot()
                            .into_iter()
                            .find(|model| model.provider.0 == provider && model.id == id)
                    });
            let scoped = self.session().scoped_models();
            let models = self.session().model_runtime().get_available_snapshot();
            let selector = Rc::new(
                crate::modes::interactive::components::model_selector::ModelSelectorComponent::new(
                    crate::modes::interactive::components::model_selector::ModelSelectorConfig {
                        current_model: self.session().model(),
                        models,
                        scoped_models: scoped.into_iter().map(|scoped| scoped.model).collect(),
                        default_model,
                        initial_search: initial_search.map(str::to_owned),
                        refresh_status: None,
                    },
                    crate::modes::interactive::components::model_selector::ModelSelectorCallbacks {
                        on_select: Rc::new(move |model| {
                            let _ = commands.send(FrameCommand::SetModel {
                                model,
                                persist: false,
                            });
                        }),
                        on_select_as_default: Some(Rc::new(move |model| {
                            let _ = commands_default.send(FrameCommand::SetModel {
                                model,
                                persist: true,
                            });
                        })),
                        on_cancel: Rc::new(move || cancel_done()),
                    },
                ),
            );
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }

    /// Refresh the footer's provider count.
    pub(crate) async fn update_available_provider_count(&self) {
        let models = if self.session().scoped_models().is_empty() {
            self.session().model_runtime().get_available_snapshot()
        } else {
            self.session()
                .scoped_models()
                .into_iter()
                .map(|scoped| scoped.model)
                .collect()
        };
        let providers: std::collections::BTreeSet<String> =
            models.into_iter().map(|model| model.provider.0).collect();
        self.footer_data_provider
            .set_available_provider_count(providers.len());
    }

    /// Warn once when Anthropic subscription auth is in use.
    pub async fn maybe_warn_about_anthropic_subscription_auth(&self, model: Option<&Model>) {
        let model = model.cloned().or_else(|| self.session().model());
        let Some(model) = model else {
            return;
        };
        let disabled = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_warnings()
            .get("anthropicExtraUsage")
            .and_then(serde_json::Value::as_bool)
            == Some(false);
        if disabled || self.anthropic_subscription_warning_shown.get() {
            return;
        }
        if model.provider.0 != "anthropic" {
            return;
        }
        let session = self.session();
        let runtime = session.model_runtime();
        let oauth = runtime
            .check_auth("anthropic", None)
            .await
            .ok()
            .flatten()
            .is_some_and(|check| check.auth_type == AuthType::OAuth);
        if !oauth {
            let api_key = runtime
                .get_auth("anthropic", None)
                .await
                .ok()
                .flatten()
                .and_then(|result| result.auth.api_key);
            let subscription_key = api_key.is_some_and(|key| key.starts_with("sk-ant-oat"));
            if !subscription_key {
                return;
            }
        }
        self.anthropic_subscription_warning_shown.set(true);
        self.show_warning(
            "Anthropic subscription auth is active. Third-party harness usage draws from extra usage and is billed per token, not your Claude plan limits. Manage extra usage at https://claude.ai/settings/usage. Disable this warning in /settings.",
        );
    }

    /// The Daxnuts easter egg.
    pub(crate) fn check_daxnuts_easter_egg(&self, model: &Model) {
        if model.provider.0 == "opencode" && model.id.to_lowercase().contains("kimi-k2.5") {
            self.handle_daxnuts();
        }
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show the scoped-model picker, upstream's `showModelsSelector`.
    pub(crate) async fn show_models_selector(&self) {
        let available = self.session().model_runtime().get_available_snapshot();
        let configured = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_enabled_models();
        let session_scoped = self.session().scoped_models();
        let initial: Option<Vec<String>> = if session_scoped.is_empty() {
            configured.map(|patterns| {
                crate::model_resolver::resolve_model_scope_from_models(&patterns, &available)
                    .scoped_models
                    .into_iter()
                    .map(|scoped| format!("{}/{}", scoped.model.provider.0, scoped.model.id))
                    .collect()
            })
        } else {
            Some(
                session_scoped
                    .iter()
                    .map(|scoped| format!("{}/{}", scoped.model.provider.0, scoped.model.id))
                    .collect(),
            )
        };
        let refresh = crate::modes::interactive::model_catalog_refresh::refresh_model_catalogs(
            self.session().model_runtime(),
            &tokio_util::sync::CancellationToken::new(),
        )
        .await;
        let refresh_status = match refresh {
            Ok(result) if result.errors.is_empty() => Some("Model catalogs refreshed.".to_owned()),
            Ok(_) => Some("Some model catalogs could not be refreshed.".to_owned()),
            Err(_) => Some("Model refresh timed out; showing cached models.".to_owned()),
        };
        let models = self.session().model_runtime().get_available_snapshot();
        let commands = self.commands();
        let commands_persist = commands.clone();
        let settings = self.settings();
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let selector = Rc::new(
                crate::modes::interactive::components::scoped_models_selector::ScopedModelsSelectorComponent::new(
                    crate::modes::interactive::components::scoped_models_selector::ModelsConfig {
                        all_models: models.clone(),
                        enabled_model_ids: initial.clone(),
                        refresh_status: refresh_status.clone(),
                    },
                    crate::modes::interactive::components::scoped_models_selector::ModelsCallbacks {
                        on_change: Rc::new({
                            let commands = commands.clone();
                            move |enabled| {
                                let _ = commands.send(FrameCommand::SetScopedModels(enabled));
                            }
                        }),
                        on_persist: Rc::new({
                            let commands = commands_persist.clone();
                            move |enabled| {
                                let _ = commands.send(FrameCommand::PersistScopedModels(enabled));
                            }
                        }),
                        on_cancel: Rc::new(move || cancel_done()),
                    },
                ),
            );
            let _ = &settings;
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }

    /// Apply a scoped-model selection to the session.
    pub(crate) fn apply_scoped_models(&self, enabled: Option<Vec<String>>) {
        let available = self.session().model_runtime().get_available_snapshot();
        let has_available = enabled.as_ref().is_some_and(|ids| {
            ids.iter().any(|id| {
                available
                    .iter()
                    .any(|model| format!("{}/{}", model.provider.0, model.id) == *id)
            })
        });
        let all_enabled = enabled.as_ref().is_some_and(|ids| {
            ids.len() == available.len()
                && available
                    .iter()
                    .all(|model| ids.contains(&format!("{}/{}", model.provider.0, model.id)))
        });
        match enabled {
            Some(ids) if has_available && !all_enabled => {
                let scoped =
                    crate::model_resolver::resolve_model_scope_from_models(&ids, &available)
                        .scoped_models;
                self.session().set_scoped_models(scoped);
            }
            _ => self.session().set_scoped_models(Vec::new()),
        }
        self.tui.request_render(false);
    }

    /// Persist a scoped-model selection.
    pub(crate) fn persist_scoped_models(&self, enabled: Option<Vec<String>>) {
        let available = self.session().model_runtime().get_available_snapshot();
        let all_enabled = enabled.as_ref().is_some_and(|ids| {
            ids.len() == available.len()
                && available
                    .iter()
                    .all(|model| ids.contains(&format!("{}/{}", model.provider.0, model.id)))
        });
        let patterns = enabled.filter(|_| !all_enabled);
        self.settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_enabled_models(patterns.as_deref());
        self.show_status("Model selection saved to settings");
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show the session-tree selector, upstream's `showTreeSelector`.
    pub async fn show_tree_selector(&self, initial_selected: Option<&str>) {
        let tree = crate::core::agent_session::lock(&self.session().session_manager).get_tree();
        if tree.is_empty() {
            self.show_status("No entries in session");
            return;
        }
        let leaf = crate::core::agent_session::lock(&self.session().session_manager)
            .get_leaf_id()
            .map(str::to_owned);
        let filter_mode = match self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_tree_filter_mode()
        {
            "no-tools" => crate::modes::interactive::components::tree_selector::FilterMode::NoTools,
            "user-only" => {
                crate::modes::interactive::components::tree_selector::FilterMode::UserOnly
            }
            "labeled-only" => {
                crate::modes::interactive::components::tree_selector::FilterMode::LabeledOnly
            }
            "all" => crate::modes::interactive::components::tree_selector::FilterMode::All,
            _ => crate::modes::interactive::components::tree_selector::FilterMode::Default,
        };
        let commands = self.commands();
        let cancel_commands = commands.clone();
        let terminal_height = usize::from(self.tui.terminal_rows());
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let selector =
                crate::modes::interactive::components::tree_selector::TreeSelectorComponent::new(
                    &tree,
                    leaf.as_deref(),
                    terminal_height,
                    crate::modes::interactive::components::tree_selector::TreeSelectorOptions {
                        on_select: Rc::new(move |entry_id: &str| {
                            let _ = commands.send(FrameCommand::NavigateTree(entry_id.to_owned()));
                        }),
                        on_cancel: Rc::new(move || cancel_done()),
                        on_label_change: None,
                        initial_selected_id: initial_selected.map(str::to_owned),
                        initial_filter_mode: Some(filter_mode),
                    },
                );
            let _ = &cancel_commands;
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }

    /// Show the user-message fork picker.
    pub(crate) fn show_user_message_selector(&self) {
        let messages = self.session().user_messages_for_forking();
        if messages.is_empty() {
            self.show_status("No messages to fork from");
            return;
        }
        let initial = messages.last().map(|(entry_id, _)| entry_id.clone());
        let commands = self.commands();
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let selector =
                crate::modes::interactive::components::user_message_selector::UserMessageSelectorComponent::new(
                    messages
                        .iter()
                        .map(|(entry_id, text)| {
                            crate::modes::interactive::components::user_message_selector::UserMessageItem {
                                id: entry_id.clone(),
                                text: text.clone(),
                            }
                        })
                        .collect(),
                    Rc::new(move |entry_id: &str| {
                        let _ = commands.send(FrameCommand::ForkEntry(entry_id.to_owned()));
                    }),
                    Rc::new(move || cancel_done()),
                    initial.as_deref(),
                );
            SelectorParts {
                focus: selector.get_message_list() as Rc<dyn TuiComponent>,
                component: Rc::new(selector),
                dispose: None,
            }
        });
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show the session picker, upstream's `showSessionSelector`.
    pub(crate) async fn show_session_selector(&self) {
        let (cwd, session_dir, session_file, uses_default) = {
            let session = self.session();
            let manager = crate::core::agent_session::lock(&session.session_manager);
            (
                manager.cwd().to_owned(),
                manager.session_dir().to_owned(),
                manager.session_file().map(str::to_owned),
                manager.uses_default_session_dir(),
            )
        };
        let current = crate::session_manager::SessionManager::list(&cwd, Some(&session_dir), None)
            .await
            .unwrap_or_default();
        let all = if uses_default {
            crate::session_manager::SessionManager::list_all(None).await
        } else {
            crate::session_manager::SessionManager::list_all_from_dir(&session_dir, None)
                .await
                .unwrap_or_default()
        };
        let commands = self.commands();
        let request_render: Rc<dyn Fn()> = {
            let tui = Rc::clone(&self.tui);
            Rc::new(move || tui.request_render(false))
        };
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let refresh: Rc<dyn Fn()> = Rc::new(|| {});
            let selector = crate::modes::interactive::components::session_selector::SessionSelectorComponent::new(
                crate::modes::interactive::components::session_selector::SessionSelectorOptions {
                    current_sessions: current.clone(),
                    current_session_path: session_file.clone(),
                    on_select: Rc::new({
                        let commands = commands.clone();
                        move |path: &str| {
                            let _ = commands.send(FrameCommand::SwitchSession(path.to_owned()));
                        }
                    }),
                    on_cancel: Rc::new(move || cancel_done()),
                    on_request_all_sessions: Rc::clone(&refresh),
                    on_refresh_sessions: Rc::clone(&refresh),
                    request_render: Rc::clone(&request_render),
                    rename_session: Some(Rc::new(|path: &str, name: &str| {
                        if let Ok(mut manager) =
                            crate::session_manager::SessionManager::open(path, None, None)
                        {
                            let _ = manager.append_session_info(name);
                        }
                    })),
                    show_rename_hint: Some(true),
                },
            );
            selector.set_all_sessions(all.clone());
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }

    /// Show the project-trust selector.
    pub(crate) fn show_trust_selector(&self) {
        let cwd = crate::core::agent_session::lock(&self.session().session_manager)
            .cwd()
            .to_owned();
        let agent_dir = self
            .runtime_host
            .borrow()
            .as_ref()
            .map(|runtime| runtime.services().agent_dir.clone())
            .unwrap_or_default();
        let trust_store =
            std::sync::Arc::new(crate::trust_manager::ProjectTrustStore::new(&agent_dir));
        let saved = trust_store.get_entry(&cwd).ok().flatten();
        let trusted = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_project_trusted();
        let commands = self.commands();
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let store = std::sync::Arc::clone(&trust_store);
            let done_for_select = Rc::clone(&done);
            let selector = Rc::new(
                crate::modes::interactive::components::trust_selector::TrustSelectorComponent::new(
                    crate::modes::interactive::components::trust_selector::TrustSelectorOptions {
                        cwd: cwd.clone(),
                        saved_decision: saved.clone(),
                        project_trusted: trusted,
                        on_select: Rc::new(move |selection| {
                            let _ = store.set_many(&selection.updates);
                            let _ = commands.send(FrameCommand::NotifyTrustSaved {
                                trusted: selection.trusted,
                            });
                            done_for_select();
                        }),
                        on_cancel: Rc::new(move || cancel_done()),
                    },
                ),
            );
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Fork one entry from a user message.
    pub(crate) async fn fork_entry(&self, entry_id: &str, at: bool) {
        let mut runtime = self.take_runtime();
        let entry_id = entry_id.to_owned();
        let options = ForkOptions {
            position: at.then_some(crate::extensions::reasons::ForkPosition::At),
            ..ForkOptions::default()
        };
        let (runtime, result) = self
            .drive(async move {
                let outcome = runtime.fork(&entry_id, options).await;
                (runtime, outcome)
            })
            .await;
        self.put_runtime(runtime);
        match result {
            Ok(outcome) => {
                if outcome.cancelled {
                    return;
                }
                self.rebind_current_session(true).await;
                self.editor()
                    .set_text(outcome.selected_text.as_deref().unwrap_or_default());
                self.show_status(if at {
                    "Cloned to new session"
                } else {
                    "Forked to new session"
                });
            }
            Err(error) => self.show_error(&error.to_string()),
        }
    }

    /// Navigate the session tree to one entry, upstream's tree-selector
    /// callback: ask about summarization, abort a live response, recheck the
    /// operation guards, then replace the transcript.
    pub async fn navigate_tree(&self, entry_id: &str) {
        let leaf = crate::core::agent_session::lock(&self.session().session_manager)
            .get_leaf_id()
            .map(str::to_owned);
        if leaf.as_deref() == Some(entry_id) {
            self.dispose_active_selector();
            self.show_status("Already at this point");
            return;
        }
        self.dispose_active_selector();

        let mut wants_summary = false;
        let mut custom_instructions = None;
        let skip_prompt = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_branch_summary_skip_prompt();
        if !skip_prompt {
            loop {
                let Some(choice) = self
                    .ask_extension_select(
                        "Summarize branch?",
                        vec![
                            "No summary".to_owned(),
                            "Summarize".to_owned(),
                            "Summarize with custom prompt".to_owned(),
                        ],
                    )
                    .await
                else {
                    self.show_tree_selector(Some(entry_id)).await;
                    return;
                };
                wants_summary = choice != "No summary";
                if choice == "Summarize with custom prompt" {
                    let Some(instructions) = self
                        .ask_extension_editor("Custom summarization instructions")
                        .await
                    else {
                        continue;
                    };
                    custom_instructions = Some(instructions);
                }
                break;
            }
        }

        if self.session().is_streaming() {
            self.restore_queued_messages_to_editor(false, None);
            self.session().abort().await;
        }
        if self.session().is_compacting() {
            self.show_error(
                "Wait for the current compaction or tree navigation to finish before navigating the session tree.",
            );
            return;
        }

        let original_on_escape = self.editor().on_escape();
        let mut showing_summary_indicator = false;
        if wants_summary {
            let session = self.session();
            self.default_editor.set_on_escape(Some(Rc::new(move || {
                session.abort_branch_summary();
            })));
            self.chat_container.add_child(Rc::new(Spacer::new()));
            let indicator = self.build_status(
                crate::modes::interactive::components::status_indicator::StatusIndicatorKind::BranchSummary,
                "Summarizing branch...".to_owned(),
                None,
            );
            self.show_status_indicator(indicator);
            showing_summary_indicator = true;
            self.tui.request_render(false);
        }

        let result = self
            .session()
            .navigate_tree(
                entry_id,
                NavigateTreeOptions {
                    summarize: wants_summary,
                    custom_instructions,
                    ..NavigateTreeOptions::default()
                },
            )
            .await;
        match result {
            Ok(outcome) => {
                if outcome.aborted {
                    self.show_status("Branch summarization cancelled");
                    self.show_tree_selector(Some(entry_id)).await;
                } else if outcome.cancelled {
                    self.show_status("Navigation cancelled");
                } else {
                    self.chat_container.clear();
                    self.assistant_rows.borrow_mut().clear();
                    self.render_initial_messages();
                    if let Some(text) = outcome.editor_text
                        && self.editor().get_text().trim().is_empty()
                    {
                        self.editor().set_text(&text);
                    }
                    self.show_status("Navigated to selected point");
                    self.flush_compaction_queue(false);
                }
            }
            Err(error) => self.show_error(&error),
        }
        if showing_summary_indicator {
            self.clear_status_indicator(Some(
                crate::modes::interactive::components::status_indicator::StatusIndicatorKind::BranchSummary,
            ));
        }
        self.default_editor.set_on_escape(original_on_escape);
    }

    /// Show one extension selector and await its answer while the run loop
    /// keeps servicing the TUI, upstream's `await this.showExtensionSelector`.
    async fn ask_extension_select(&self, title: &str, options: Vec<String>) -> Option<String> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let sender = Rc::new(std::cell::RefCell::new(Some(sender)));
        let complete = Rc::new(move |value: Option<String>| {
            if let Some(sender) = sender.borrow_mut().take() {
                let _ = sender.send(value);
            }
        });
        self.show_extension_selector(title, options, None, complete);
        let mut receiver = receiver;
        self.drive(async {
            loop {
                match receiver.try_recv() {
                    Ok(value) => return value,
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => return None,
                }
            }
        })
        .await
    }

    /// Show the extension editor and await its text, upstream's
    /// `await this.showExtensionEditor`.
    async fn ask_extension_editor(&self, title: &str) -> Option<String> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let sender = Rc::new(std::cell::RefCell::new(Some(sender)));
        let complete = Rc::new(move |value: Option<String>| {
            if let Some(sender) = sender.borrow_mut().take() {
                let _ = sender.send(value);
            }
        });
        self.show_extension_editor(title, None, complete);
        let mut receiver = receiver;
        self.drive(async {
            loop {
                match receiver.try_recv() {
                    Ok(value) => return value,
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => return None,
                }
            }
        })
        .await
    }

    /// Clone the session at its current leaf.
    pub async fn handle_clone_command(&self) {
        let leaf = crate::core::agent_session::lock(&self.session().session_manager)
            .get_leaf_id()
            .map(str::to_owned);
        let Some(leaf) = leaf else {
            self.show_status("Nothing to clone yet");
            return;
        };
        self.fork_entry(&leaf, true).await;
    }

    /// Switch to another session file.
    pub(crate) async fn switch_session(
        &self,
        path: &str,
    ) -> Result<(), crate::core::agent_session_runtime::AgentSessionRuntimeError> {
        let ui = Arc::clone(&self.ui_context);
        let project_trust: crate::core::agent_session_runtime::ProjectTrustContextFactory =
            Arc::new(move |_cwd| {
                Some(Arc::new(TrustContext {
                    ui: Arc::clone(&ui),
                })
                    as Arc<dyn crate::trust_manager::ProjectTrustSelector>)
            });
        let mut runtime = self.take_runtime();
        let path = path.to_owned();
        let path_for_retry = path.clone();
        let (runtime, result) = self
            .drive(async move {
                let outcome = runtime
                    .switch_session(
                        &path,
                        SwitchSessionOptions {
                            project_trust_context_factory: Some(project_trust),
                            ..SwitchSessionOptions::default()
                        },
                    )
                    .await;
                (runtime, outcome)
            })
            .await;
        self.put_runtime(runtime);
        match result {
            Ok(outcome) => {
                if !outcome.cancelled {
                    self.rebind_current_session(true).await;
                    self.show_status("Resumed session");
                }
                Ok(())
            }
            Err(error) => {
                if let crate::core::agent_session_runtime::AgentSessionRuntimeError::MissingSessionCwd(missing) = &error {
                    let fallback = missing.issue.fallback_cwd.clone();
                    self.show_warning(&crate::session_cwd::format_missing_session_cwd_prompt(
                        &missing.issue,
                    ));
                    let mut runtime = self.take_runtime();
                    let path = path_for_retry.clone();
                    let (runtime, result) = self
                        .drive(async move {
                            let outcome = runtime
                                .switch_session(
                                    &path,
                                    SwitchSessionOptions {
                                        cwd_override: Some(fallback),
                                        ..SwitchSessionOptions::default()
                                    },
                                )
                                .await;
                            (runtime, outcome)
                        })
                        .await;
                    self.put_runtime(runtime);
                    if result.is_ok() {
                        self.rebind_current_session(true).await;
                        self.show_status("Resumed session in current cwd");
                        return Ok(());
                    }
                }
                self.show_error(&error.to_string());
                Err(error)
            }
        }
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Report a startup submission while initialization is still running.
    pub fn handle_startup_submit(&self, text: &str) {
        self.editor().set_text(text);
        self.show_status("Startup is still in progress");
    }

    /// Report a fatal runtime error, upstream's `handleFatalRuntimeError`.
    ///
    /// Restatement: the port has no process exit; the frame reports the error,
    /// stops the theme watcher and TUI, and asks the loop to exit.
    pub(crate) async fn handle_fatal_runtime_error(&self, prefix: &str, error: &str) {
        self.show_error(&format!("{prefix}: {error}"));
        crate::modes::interactive::theme::theme::stop_theme_watcher();
        self.stop(
            self.settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get_fullscreen_exit_output(),
        );
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Save an implicit project-trust decision after reload.
    pub(crate) fn maybe_save_implicit_project_trust_after_reload(&self) -> bool {
        let cwd = crate::core::agent_session::lock(&self.session().session_manager)
            .cwd()
            .to_owned();
        if self.auto_trust_on_reload_cwd.borrow().as_deref() != Some(cwd.as_str()) {
            return false;
        }
        let trusted = self
            .settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_project_trusted();
        if !trusted || !crate::trust_manager::has_trust_requiring_project_resources(&cwd) {
            return false;
        }
        let agent_dir = self
            .runtime_host
            .borrow()
            .as_ref()
            .map(|runtime| runtime.services().agent_dir.clone())
            .unwrap_or_default();
        let store = crate::trust_manager::ProjectTrustStore::new(&agent_dir);
        if store.get(&cwd).ok().flatten().is_some() {
            *self.auto_trust_on_reload_cwd.borrow_mut() = None;
            return false;
        }
        match store.set(&cwd, Some(true)) {
            Ok(()) => {
                *self.auto_trust_on_reload_cwd.borrow_mut() = None;
                true
            }
            Err(error) => {
                self.show_warning(&format!(
                    "Could not save project trust after reload: {error}"
                ));
                false
            }
        }
    }

    /// Apply the fullscreen scrollbar setting.
    pub(crate) fn apply_fullscreen_scrollbar_setting(&self) {
        if let Some(transcript) = self.transcript_scroll_view.borrow().as_ref() {
            transcript.set_scrollbar(
                self.settings()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_fullscreen_scrollbar(),
            );
        }
    }

    /// Extract the text of a user message.
    pub(crate) fn get_user_message_text(&self, message: &pi_ai::types::Message) -> String {
        let pi_ai::types::Message::User(user) = message else {
            return String::new();
        };
        pi_ai::utils::text::content_text(&user.content)
    }

    /// All queued messages, session and compaction queues combined.
    #[must_use]
    pub fn get_all_queued_messages(&self) -> (Vec<String>, Vec<String>) {
        let mut steering = self.session().steering_messages();
        let mut follow_up = self.session().follow_up_messages();
        for message in self.compaction_queued_messages.borrow().iter() {
            match message.mode {
                super::QueueDelivery::Steer => steering.push(message.text.clone()),
                super::QueueDelivery::FollowUp => follow_up.push(message.text.clone()),
            }
        }
        (steering, follow_up)
    }

    /// Clear both queues and return their contents.
    pub(crate) fn clear_all_queues(&self) -> (Vec<String>, Vec<String>) {
        let (mut steering, mut follow_up) = self.session().clear_queue();
        for message in std::mem::take(&mut *self.compaction_queued_messages.borrow_mut()) {
            match message.mode {
                super::QueueDelivery::Steer => steering.push(message.text),
                super::QueueDelivery::FollowUp => follow_up.push(message.text),
            }
        }
        (steering, follow_up)
    }

    /// Find provider options by id or display name.
    pub(crate) fn find_login_provider_options(
        &self,
        provider_ref: &str,
    ) -> Vec<crate::modes::interactive::components::oauth_selector::AuthSelectorProvider> {
        let normalized = provider_ref.trim().to_lowercase();
        if normalized.is_empty() {
            return Vec::new();
        }
        self.get_login_provider_options(None)
            .into_iter()
            .filter(|provider| {
                provider.id.to_lowercase() == normalized
                    || provider.name.to_lowercase() == normalized
            })
            .collect()
    }
}

struct TrustContext {
    ui: Arc<super::extensions::InteractiveUiContext>,
}

impl crate::trust_manager::ProjectTrustSelector for TrustContext {
    fn select(
        &self,
        prompt: String,
        options: Vec<String>,
    ) -> pi_ai::types::BoxedFuture<'static, Option<String>> {
        self.ui.select(prompt, options, None, None)
    }

    fn has_ui(&self) -> bool {
        true
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Show the settings selector, upstream's `showSettingsSelector`.
    pub(crate) fn show_settings_selector(&self) {
        let settings = self.settings();
        let settings = settings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let default_provider = settings.get_default_provider();
        let default_model_id = settings.get_default_model();
        let default_model = default_provider
            .zip(default_model_id)
            .map(|(provider, id)| format!("{provider}/{id}"))
            .unwrap_or_else(|| "not set".to_owned());
        let model_thinking_levels = settings
            .get_all_model_thinking_levels()
            .into_iter()
            .filter_map(|(key, value)| {
                serde_json::from_value::<ThinkingLevel>(value)
                    .ok()
                    .map(|level| (key, level))
            })
            .collect();
        let config = crate::modes::interactive::components::settings_selector::SettingsConfig {
            auto_compact: self.session().auto_compaction_enabled(),
            default_model,
            current_model: self.session().model(),
            available_default_models: self.session().model_runtime().get_available_snapshot(),
            show_images: settings.get_show_images(),
            image_width_cells: settings.get_image_width_cells(),
            auto_resize_images: settings.get_image_auto_resize(),
            block_images: settings.get_block_images(),
            enable_skill_commands: settings.get_enable_skill_commands(),
            steering_mode: steering_mode(self.session().steering_mode()),
            follow_up_mode: steering_mode(self.session().follow_up_mode()),
            transport: settings.get_transport(),
            http_idle_timeout_ms: settings.get_http_idle_timeout_ms().unwrap_or_default(),
            thinking_level: settings
                .get_default_thinking_level()
                .unwrap_or(crate::defaults::DEFAULT_THINKING_LEVEL),
            available_thinking_levels: crate::defaults::THINKING_LEVEL_OPTIONS.to_vec(),
            model_thinking_levels,
            current_theme: self
                .theme_controller
                .get_theme_selection()
                .unwrap_or_else(|| "dark".to_owned()),
            terminal_theme: self.theme_controller.get_terminal_theme(),
            available_themes: crate::modes::interactive::theme::theme::get_available_themes(),
            hide_thinking_block: self.hide_thinking_block.get(),
            mermaid_rendering_mode: settings.get_mermaid_rendering_mode(),
            show_cache_miss_notices: settings.get_show_cache_miss_notices(),
            collapse_changelog: settings.get_collapse_changelog(),
            enable_install_telemetry: settings.get_enable_install_telemetry(),
            double_escape_action: settings.get_double_escape_action().to_owned(),
            tree_filter_mode: settings.get_tree_filter_mode().to_owned(),
            show_hardware_cursor: settings.get_show_hardware_cursor(),
            editor_padding_x: settings.get_editor_padding_x(),
            output_pad: settings.get_output_pad(),
            autocomplete_max_visible: settings.get_autocomplete_max_visible(),
            quiet_startup: settings.get_quiet_startup(),
            default_project_trust: settings.get_default_project_trust(),
            clear_on_shrink: settings.get_clear_on_shrink(),
            show_terminal_progress: settings.get_show_terminal_progress(),
            tui_mode: self.tui.mode(),
            fullscreen_exit_output: settings.get_fullscreen_exit_output(),
            fullscreen_scrollbar: settings.get_fullscreen_scrollbar(),
            fullscreen_copy_on_select: settings.get_fullscreen_copy_on_select(),
            warnings: warning_settings(&settings),
        };
        drop(settings);
        let callbacks = self.settings_callbacks();
        self.show_selector(|_done| {
            let selector =
                crate::modes::interactive::components::settings_selector::SettingsSelectorComponent::new(
                    &config,
                    callbacks,
                );
            SelectorParts {
                focus: selector.get_settings_list(),
                component: Rc::new(selector),
                dispose: None,
            }
        });
    }
}

/// Map the agent-core queue mode onto the settings vocabulary.
fn steering_mode(mode: pi_agent_core::types::QueueMode) -> crate::settings_manager::QueueMode {
    match mode {
        pi_agent_core::types::QueueMode::All => crate::settings_manager::QueueMode::All,
        pi_agent_core::types::QueueMode::OneAtATime => {
            crate::settings_manager::QueueMode::OneAtATime
        }
    }
}

/// Project the settings store's warnings object.
fn warning_settings<S: SettingsStorageKind>(
    settings: &crate::settings_manager::SettingsManager<S>,
) -> crate::modes::interactive::components::settings_selector::WarningSettings {
    crate::modes::interactive::components::settings_selector::WarningSettings {
        anthropic_extra_usage: settings
            .get_warnings()
            .get("anthropicExtraUsage")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
    }
}

/// A cloneable handle the settings callbacks write through.
struct SettingsBridge<S: SettingsStorageKind> {
    slot: super::SettingsSlot<S>,
}

impl<S: SettingsStorageKind> Clone for SettingsBridge<S> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
        }
    }
}

impl<S: SettingsStorageKind> SettingsBridge<S> {
    fn with(&self, update: impl FnOnce(&mut crate::settings_manager::SettingsManager<S>)) {
        let manager = self
            .slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let mut manager = manager
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        update(&mut manager);
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    fn settings_bridge(&self) -> SettingsBridge<S> {
        SettingsBridge {
            slot: Arc::clone(&self.settings),
        }
    }

    /// Build the settings callbacks.
    fn settings_callbacks(
        &self,
    ) -> crate::modes::interactive::components::settings_selector::SettingsCallbacks {
        use crate::modes::interactive::components::settings_selector::SettingsCallbacks;
        let commands = self.commands();
        let session = self.session();
        let footer = Rc::clone(&self.footer);
        let bridge = self.settings_bridge();
        SettingsCallbacks {
            on_auto_compact_change: Rc::new(move |enabled| {
                session.set_auto_compaction_enabled(enabled);
                footer.set_auto_compact_enabled(enabled);
            }),
            on_show_images_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_show_images(enabled);
            }),
            on_image_width_cells_change: Rc::new({
                let bridge = bridge.clone();
                move |width| {
                    bridge.with(|settings| {
                        settings.set_image_width_cells(f64::from(
                            i32::try_from(width).unwrap_or_default(),
                        ));
                    });
                }
            }),
            on_auto_resize_images_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_image_auto_resize(enabled);
            }),
            on_block_images_change: closure_settings(&bridge, |settings, blocked| {
                settings.set_block_images(blocked);
            }),
            on_enable_skill_commands_change: Rc::new({
                let bridge = bridge.clone();
                let commands = commands.clone();
                move |enabled| {
                    bridge.with(|settings| settings.set_enable_skill_commands(enabled));
                    let _ = commands.send(FrameCommand::RefreshAutocomplete);
                }
            }),
            on_steering_mode_change: {
                let session = self.session();
                Rc::new(move |mode| session.set_steering_mode(unqueue_mode(mode)))
            },
            on_follow_up_mode_change: {
                let session = self.session();
                Rc::new(move |mode| session.set_follow_up_mode(unqueue_mode(mode)))
            },
            on_transport_change: closure_settings(&bridge, |settings, transport| {
                settings.set_transport(transport);
            }),
            on_http_idle_timeout_ms_change: Rc::new({
                let bridge = bridge.clone();
                move |timeout| {
                    bridge.with(|settings| {
                        settings
                            .set_http_idle_timeout_ms(f64::from(
                                i32::try_from(timeout).unwrap_or_default(),
                            ))
                            .ok();
                    });
                }
            }),
            on_model_thinking_level_change: Rc::new({
                let bridge = bridge.clone();
                let session = self.session();
                move |provider, model_id, level| {
                    bridge.with(|settings| {
                        settings.set_model_thinking_level(provider, model_id, level);
                    });
                    if session
                        .model()
                        .is_some_and(|model| model.provider.0 == provider && model.id == model_id)
                    {
                        session.set_thinking_level(level, ModelMutationOptions::default());
                    }
                }
            }),
            on_model_thinking_level_remove: Rc::new({
                let bridge = bridge.clone();
                move |provider, model_id| {
                    bridge.with(|settings| {
                        settings.remove_model_thinking_level(provider, model_id);
                    });
                }
            }),
            on_theme_change: Rc::new({
                let commands = commands.clone();
                move |setting| {
                    let _ = commands.send(FrameCommand::SetTheme(setting.to_owned()));
                }
            }),
            on_theme_preview: Some(Rc::new({
                let commands = commands.clone();
                move |name| {
                    let _ = commands.send(FrameCommand::PreviewTheme(name.to_owned()));
                }
            })),
            on_hide_thinking_block_change: closure_settings(&bridge, |settings, hidden| {
                settings.set_hide_thinking_block(hidden);
            }),
            on_mermaid_rendering_mode_change: closure_settings(&bridge, |settings, rendering| {
                settings.set_mermaid_rendering_mode(rendering);
            }),
            on_show_cache_miss_notices_change: Rc::new({
                let bridge = bridge.clone();
                let commands = commands.clone();
                move |shown| {
                    bridge.with(|settings| settings.set_show_cache_miss_notices(shown));
                    let _ = commands.send(FrameCommand::RebuildChat);
                }
            }),
            on_collapse_changelog_change: closure_settings(&bridge, |settings, collapsed| {
                settings.set_collapse_changelog(collapsed);
            }),
            on_enable_install_telemetry_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_enable_install_telemetry(enabled);
            }),
            on_double_escape_action_change: Rc::new({
                let bridge = bridge.clone();
                move |action: &str| {
                    bridge.with(|settings| settings.set_double_escape_action(action));
                }
            }),
            on_tree_filter_mode_change: Rc::new({
                let bridge = bridge.clone();
                move |filter: &str| {
                    bridge.with(|settings| settings.set_tree_filter_mode(filter));
                }
            }),
            on_show_hardware_cursor_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_show_hardware_cursor(enabled);
            }),
            on_editor_padding_x_change: Rc::new({
                let bridge = bridge.clone();
                move |padding| {
                    bridge.with(|settings| {
                        settings.set_editor_padding_x(f64::from(
                            i32::try_from(padding).unwrap_or_default(),
                        ));
                    });
                }
            }),
            on_output_pad_change: closure_settings(&bridge, |settings, padding| {
                settings.set_output_pad(padding);
            }),
            on_autocomplete_max_visible_change: Rc::new({
                let bridge = bridge.clone();
                move |visible| {
                    bridge.with(|settings| {
                        settings.set_autocomplete_max_visible(f64::from(
                            i32::try_from(visible).unwrap_or_default(),
                        ));
                    });
                }
            }),
            on_quiet_startup_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_quiet_startup(enabled);
            }),
            on_default_project_trust_change: closure_settings(&bridge, |settings, trust| {
                settings.set_default_project_trust(trust);
            }),
            on_clear_on_shrink_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_clear_on_shrink(enabled);
            }),
            on_show_terminal_progress_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_show_terminal_progress(enabled);
            }),
            on_tui_mode_change: Rc::new({
                let commands = commands.clone();
                move |mode| {
                    let _ = commands.send(FrameCommand::SetTuiMode(mode));
                }
            }),
            on_fullscreen_exit_output_change: closure_settings(&bridge, |settings, output| {
                settings.set_fullscreen_exit_output(output);
            }),
            on_fullscreen_scrollbar_change: closure_settings(&bridge, |settings, scrollbar| {
                settings.set_fullscreen_scrollbar(scrollbar);
            }),
            on_fullscreen_copy_on_select_change: closure_settings(&bridge, |settings, enabled| {
                settings.set_fullscreen_copy_on_select(enabled)
            }),
            on_warnings_change: Rc::new({
                let bridge = bridge.clone();
                move |warnings| {
                    let mut value = crate::settings_manager::Settings::new();
                    value.insert(
                        "anthropicExtraUsage".to_owned(),
                        serde_json::Value::Bool(warnings.anthropic_extra_usage),
                    );
                    bridge.with(|settings| settings.set_warnings(&value));
                }
            }),
            on_cancel: Rc::new(move || {
                let _ = commands;
            }),
        }
    }
}

/// Wrap a settings write in the callback shape.
fn closure_settings<S, T>(
    bridge: &SettingsBridge<S>,
    update: impl Fn(&mut crate::settings_manager::SettingsManager<S>, T) + 'static,
) -> Rc<dyn Fn(T)>
where
    S: SettingsStorageKind,
    T: 'static,
{
    let bridge = bridge.clone();
    Rc::new(move |value| bridge.with(|settings| update(settings, value)))
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Apply a theme setting and persist it.
    pub(crate) fn apply_theme_setting(&self, name: &str) {
        self.settings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .set_theme(name);
        self.theme_controller.set_theme_setting(name);
    }

    /// Apply a TUI mode change.
    pub(crate) fn apply_tui_mode(&self, mode: TuiMode) {
        if self.switch_tui_mode(mode, true, true) {
            self.settings()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .set_tui_mode(mode);
            if self.active_status_indicator.borrow().is_none() {
                self.status_container.clear();
            }
            self.show_status(&format!("TUI mode: {mode:?}"));
        } else {
            self.show_status("Close active overlays before changing TUI mode");
        }
    }

    /// Read the first path argument after `command`, honoring simple quotes,
    /// upstream's `getPathCommandArgument`.
    #[must_use]
    pub fn get_path_command_argument(&self, text: &str, command: &str) -> Option<String> {
        command_argument(text, command)
    }

    /// Handle `/export [path]`.
    pub(crate) async fn handle_export_command(&self, text: &str) {
        let output = command_argument(text, "/export");
        let result = match output.as_deref() {
            Some(path) if path.ends_with(".jsonl") => self
                .session()
                .export_to_jsonl(Some(path))
                .map_err(|error| error.to_string()),
            path => self
                .session()
                .export_to_html(
                    path.map(str::to_owned),
                    crate::modes::interactive::theme::theme::current_theme_name(),
                    &crate::export_html::NoThemeSource,
                )
                .map_err(|error| error.to_string()),
        };
        match result {
            Ok(path) => self.show_status(&format!("Session exported to: {path}")),
            Err(error) => self.show_error(&format!("Failed to export session: {error}")),
        }
    }

    /// Handle `/import <path>`.
    ///
    /// Restatement: the confirmation dialog is skipped; the import runs on
    /// the named path directly.
    pub async fn handle_import_command(&self, text: &str) {
        let Some(path) = command_argument(text, "/import") else {
            self.show_error("Usage: /import <path.jsonl>");
            return;
        };
        self.clear_status_indicator(None);
        let mut runtime = self.take_runtime();
        let result = self
            .drive({
                let path = path.clone();
                async move {
                    let outcome = runtime.import_from_jsonl(&path, None).await;
                    (runtime, outcome)
                }
            })
            .await;
        let (runtime, outcome) = result;
        self.put_runtime(runtime);
        match outcome {
            Ok(result) if result.cancelled => self.show_status("Import cancelled"),
            Ok(_) => {
                self.rebind_current_session(true).await;
                self.show_status(&format!("Session imported from: {path}"));
            }
            Err(error) => self.show_error(&format!("Failed to import session: {error}")),
        }
    }

    /// Handle `/share`.
    ///
    /// Restatement: the share ladder needs process and HTTP seams the frame
    /// does not own yet; it reports unavailability instead of half-running.
    pub(crate) async fn handle_share_command(&self) {
        self.show_error("Session sharing is not wired in this build yet.");
    }

    /// Handle `/name`.
    pub(crate) fn handle_name_command(&self, text: &str) {
        let name = text.trim_start_matches("/name").trim();
        if name.is_empty() {
            let current = crate::core::agent_session::lock(&self.session().session_manager)
                .get_session_name();
            match current {
                Some(current) => self.show_status(&format!("Session name: {current}")),
                None => self.show_warning("Usage: /name <name>"),
            }
            return;
        }
        self.session().set_session_name(name);
        let actual =
            crate::core::agent_session::lock(&self.session().session_manager).get_session_name();
        if actual.as_deref() != Some(name) {
            self.show_warning(&format!(
                "Session name was normalized from {name:?} to {:?}",
                actual.clone().unwrap_or_default()
            ));
        }
        self.show_status(&format!(
            "Session name set: {}",
            actual.unwrap_or_else(|| name.to_owned())
        ));
    }

    /// Handle `/session`.
    pub(crate) fn handle_session_command(&self) {
        let info = self.session_info_markdown();
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container
            .add_child(Rc::new(Text::with_padding(info, 1, 0)));
        self.tui.request_render(false);
    }
}

/// Read the first path argument after `command`, honoring simple quotes.
fn command_argument(text: &str, command: &str) -> Option<String> {
    if text == command || !text.starts_with(&format!("{command} ")) {
        return None;
    }
    let args = text[command.len() + 1..].trim_start();
    if args.is_empty() {
        return None;
    }
    let first = args.chars().next()?;
    if first == '"' || first == '\'' {
        let closing = args[1..].find(first)?;
        return Some(args[1..closing + 1].to_owned());
    }
    Some(args.split_whitespace().next().unwrap_or(args).to_owned())
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Handle `/changelog`.
    pub(crate) fn handle_changelog_command(&self) {
        let entries =
            crate::utils::changelog::parse_changelog(&crate::config::get_changelog_path());
        let markdown = if entries.is_empty() {
            "No changelog entries found.".to_owned()
        } else {
            entries
                .iter()
                .rev()
                .map(|entry| {
                    crate::utils::changelog::normalize_changelog_links(
                        &entry.content,
                        &format!("{}.{}.{}", entry.major, entry.minor, entry.patch),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        };
        let border = crate::modes::interactive::components::dynamic_border::DynamicBorder::themed();
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(border));
        self.chat_container.add_child(Rc::new(Text::with_padding(
            bold(&theme().fg("accent", "What's New")),
            1,
            0,
        )));
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(Markdown::new(
            markdown,
            1,
            1,
            self.markdown_theme_with_settings(),
        )));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::dynamic_border::DynamicBorder::themed(),
        ));
        self.tui.request_render(false);
    }

    /// Handle `/hotkeys`.
    pub(crate) fn handle_hotkeys_command(&self) {
        let rows = [
            "app.interrupt",
            "app.clear",
            "app.exit",
            "app.suspend",
            "app.thinking.cycle",
            "app.model.cycleForward",
            "app.model.select",
            "app.tools.expand",
            "app.thinking.toggle",
            "app.editor.external",
            "app.message.copy",
            "app.message.followUp",
            "app.message.dequeue",
            "app.clipboard.pasteImage",
        ]
        .iter()
        .map(|action| {
            format!(
                "| `{}` | {action} |",
                crate::modes::interactive::components::keybinding_hints::key_display_text(action)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::dynamic_border::DynamicBorder::themed(),
        ));
        self.chat_container.add_child(Rc::new(Text::with_padding(
            bold(&theme().fg("accent", "Keyboard Shortcuts")),
            1,
            0,
        )));
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(Markdown::new(
            format!("**App**\n| Key | Action |\n|-----|--------|\n{rows}"),
            1,
            1,
            self.markdown_theme_with_settings(),
        )));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::dynamic_border::DynamicBorder::themed(),
        ));
        self.tui.request_render(false);
    }

    /// Handle `/new`.
    pub(crate) async fn handle_clear_command(&self) {
        self.clear_status_indicator(None);
        let mut runtime = self.take_runtime();
        let result = self
            .drive(async move {
                let outcome = runtime.new_session(NewSessionRequest::default()).await;
                (runtime, outcome)
            })
            .await;
        let (runtime, outcome) = result;
        self.put_runtime(runtime);
        match outcome {
            Ok(outcome) if outcome.cancelled => {}
            Ok(_) => {
                self.rebind_current_session(true).await;
                self.chat_container.add_child(Rc::new(Spacer::new()));
                self.chat_container.add_child(Rc::new(Text::with_padding(
                    theme().fg("accent", "✓ New session started"),
                    1,
                    1,
                )));
                self.tui.request_render(false);
            }
            Err(error) => self.show_error(&format!("Failed to create session: {error}")),
        }
    }

    /// Handle `/reload`.
    pub(crate) async fn handle_reload_command(&self) {
        if self.session().is_streaming() {
            self.show_warning("Wait for the current response to finish before reloading.");
            return;
        }
        self.reset_extension_ui();
        self.session().reload(None).await;
        crate::modes::interactive::theme::theme::set_registered_themes(
            super::load_registered_themes(&self.session()),
        )
        .ok();
        self.apply_runtime_settings();
        self.theme_controller.apply_from_settings();
        self.setup_autocomplete_provider();
        self.bind_current_session_extensions().await;
        self.show_loaded_resources(true);
        self.show_status(
            "Reloaded keybindings, extensions, skills, prompts, themes, and context files",
        );
    }

    /// Handle `/debug`.
    pub(crate) fn handle_debug_command(&self) {
        let width = usize::from(self.tui.terminal_columns());
        let height = usize::from(self.tui.terminal_rows());
        let lines = self.tui.render_children(width);
        let debug_path = crate::config::get_debug_log_path();
        let mut data = format!(
            "Debug output at {}\nTerminal: {width}x{height}\nTotal lines: {}\n\n",
            now_timestamp(),
            lines.len()
        );
        for (index, line) in lines.iter().enumerate() {
            data.push_str(&format!(
                "[{index}] (w={}) {}\n",
                pi_tui::utils::visible_width(line),
                serde_json::to_string(line).unwrap_or_default()
            ));
        }
        data.push_str("\n=== Agent messages (JSONL) ===\n");
        for message in &self.session().agent().state().messages {
            data.push_str(&serde_json::to_string(message).unwrap_or_default());
            data.push('\n');
        }
        if let Some(parent) = debug_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        match std::fs::write(&debug_path, data) {
            Ok(()) => self.show_status(&format!("✓ Debug log written\n{}", debug_path.display())),
            Err(error) => self.show_error(&format!("Failed to write debug log: {error}")),
        }
    }

    /// `/arminsayshi`.
    pub(crate) fn handle_armin_says_hi(&self) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::armin::ArminComponent::new(
                self.render_request(),
            ),
        ));
        self.tui.request_render(false);
    }

    /// `/dementedelves`.
    pub(crate) fn handle_demented_delves(&self) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::earendil_announcement::EarendilAnnouncementComponent::new(),
        ));
        self.tui.request_render(false);
    }

    /// The Daxnuts row.
    pub(crate) fn handle_daxnuts(&self) {
        self.chat_container.add_child(Rc::new(Spacer::new()));
        self.chat_container.add_child(Rc::new(
            crate::modes::interactive::components::daxnuts::DaxnutsComponent::new(
                self.render_request(),
            ),
        ));
        self.tui.request_render(false);
    }
}

/// Map the settings queue mode back onto the agent-core vocabulary.
fn unqueue_mode(mode: crate::settings_manager::QueueMode) -> pi_agent_core::types::QueueMode {
    match mode {
        crate::settings_manager::QueueMode::All => pi_agent_core::types::QueueMode::All,
        crate::settings_manager::QueueMode::OneAtATime => {
            pi_agent_core::types::QueueMode::OneAtATime
        }
    }
}

/// The current wall clock as a millisecond stamp for the debug log.
fn now_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .to_string()
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// The provider options the login selector offers.
    fn get_login_provider_options(
        &self,
        auth_type: Option<AuthType>,
    ) -> Vec<crate::modes::interactive::components::oauth_selector::AuthSelectorProvider> {
        use crate::modes::interactive::components::oauth_selector::{
            AuthSelectorMethod, AuthSelectorProvider,
        };
        let session = self.session();
        let runtime = session.model_runtime();
        let mut options = Vec::new();
        for provider in runtime.get_providers() {
            let status = runtime.get_provider_auth_status(provider.id());
            let check = status.configured.then(|| pi_ai::auth::types::AuthCheck {
                source: status.label.clone(),
                auth_type: if runtime.is_using_oauth(provider.id()) {
                    AuthType::OAuth
                } else {
                    AuthType::ApiKey
                },
            });
            if auth_type != Some(AuthType::ApiKey)
                && let Some(oauth) = provider.auth().oauth.clone()
            {
                options.push(AuthSelectorProvider {
                    id: provider.id().to_owned(),
                    name: provider.name().to_owned(),
                    auth_type: AuthType::OAuth,
                    method: Some(AuthSelectorMethod::OAuth(oauth)),
                    status: check.clone(),
                });
            }
            if auth_type != Some(AuthType::OAuth)
                && let Some(api_key) = provider.auth().api_key.clone()
            {
                options.push(AuthSelectorProvider {
                    id: provider.id().to_owned(),
                    name: provider.name().to_owned(),
                    auth_type: AuthType::ApiKey,
                    method: Some(AuthSelectorMethod::ApiKey(api_key)),
                    status: check,
                });
            }
        }
        options.sort_by(|left, right| left.name.cmp(&right.name));
        options
    }

    /// Handle `/login [provider]`, upstream's `handleLoginCommand`.
    pub(crate) async fn handle_login_command(&self, provider_ref: Option<&str>) {
        let Some(provider_ref) = provider_ref
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            self.show_oauth_selector(true).await;
            return;
        };
        let matches = self.find_login_provider_options(provider_ref);
        if matches.len() == 1 {
            self.start_provider_login(matches[0].clone()).await;
            return;
        }
        self.show_oauth_selector(true).await;
    }

    /// Show the provider selector for login or logout.
    pub(crate) async fn show_oauth_selector(&self, login: bool) {
        use crate::modes::interactive::components::oauth_selector::{
            AuthSelectorProvider, OAuthSelectorComponent, OAuthSelectorMode,
        };
        let providers: Vec<AuthSelectorProvider> = if login {
            self.get_login_provider_options(None)
        } else {
            let credentials = self
                .session()
                .model_runtime()
                .list_credentials(None)
                .await
                .unwrap_or_default();
            let session = self.session();
            let runtime = session.model_runtime();
            credentials
                .into_iter()
                .map(|credential| {
                    let name = runtime.get_provider(&credential.provider_id).map_or_else(
                        || credential.provider_id.clone(),
                        |provider| provider.name().to_owned(),
                    );
                    AuthSelectorProvider {
                        id: credential.provider_id.clone(),
                        name,
                        auth_type: credential.auth_type,
                        method: None,
                        status: Some(pi_ai::auth::types::AuthCheck {
                            source: Some("stored credential".to_owned()),
                            auth_type: credential.auth_type,
                        }),
                    }
                })
                .collect()
        };
        if providers.is_empty() {
            self.show_status(if login {
                "No login providers available."
            } else {
                "No stored credentials to remove. /logout only removes credentials saved by /login."
            });
            return;
        }
        let commands = self.commands();
        let logout = !login;
        self.show_selector(|done| {
            let cancel_done = Rc::clone(&done);
            let selector_providers = providers.clone();
            let selector = Rc::new(OAuthSelectorComponent::new(
                if login {
                    OAuthSelectorMode::Login
                } else {
                    OAuthSelectorMode::Logout
                },
                providers.clone(),
                Rc::new(move |provider_id: String, auth_type: AuthType| {
                    if logout {
                        let _ = commands.send(FrameCommand::LogoutProvider(provider_id));
                    } else if let Some(provider) = selector_providers.iter().find(|provider| {
                        provider.id == provider_id && provider.auth_type == auth_type
                    }) {
                        let _ =
                            commands.send(FrameCommand::HandleStartProviderLogin(provider.clone()));
                    }
                }),
                Rc::new(move || cancel_done()),
                None,
            ));
            SelectorParts {
                focus: Rc::clone(&selector) as Rc<dyn TuiComponent>,
                component: selector,
                dispose: None,
            }
        });
    }
}

impl<S: SettingsStorageKind> InteractiveMode<S> {
    /// Start one provider login, upstream's `startProviderLogin`.
    pub(crate) async fn start_provider_login(
        &self,
        provider: crate::modes::interactive::components::oauth_selector::AuthSelectorProvider,
    ) {
        use crate::modes::interactive::components::login_dialog::LoginDialogComponent;
        let dialog = Rc::new(LoginDialogComponent::new(
            Rc::clone(&self.tui),
            &provider.id,
            Rc::new(|_success, _message| {}),
            Some(&provider.name),
            None,
        ));
        *self.login_dialog.borrow_mut() = Some(Rc::clone(&dialog));
        self.editor_container.clear();
        self.editor_container
            .add_child(Rc::clone(&dialog) as Rc<dyn TuiComponent>);
        self.tui.set_focus(Some(dialog as Rc<dyn TuiComponent>));
        self.tui.request_render(false);
        let runtime = self.session().model_runtime().clone();
        let ui = Arc::clone(&self.ui_context);
        let notify_ui = Arc::clone(&self.ui_context);
        let commands = self.commands();
        let provider_id = provider.id.clone();
        let provider_name = provider.name.clone();
        let auth_type = provider.auth_type;
        tokio::spawn(async move {
            let interaction = pi_ai::auth::types::AuthInteraction {
                signal: None,
                prompt: Arc::new(move |prompt: pi_ai::auth::types::AuthPrompt| {
                    let ui = Arc::clone(&ui);
                    Box::pin(async move {
                        let (reply, receiver) = tokio::sync::oneshot::channel();
                        ui.send_auth_prompt(prompt, reply);
                        receiver
                            .await
                            .unwrap_or_else(|_| Err("Login cancelled".to_owned()))
                            .map_err(|_| pi_ai::utils::abort::AbortError)
                    })
                }),
                notify: Arc::new(move |event| notify_ui.send_auth_notify(event)),
            };
            let result = runtime
                .login(&provider_id, auth_type, interaction)
                .await
                .map(|_| ())
                .map_err(|error| error.to_string());
            let _ = commands.send(FrameCommand::LoginFinished {
                provider_name,
                auth_type,
                result,
            });
        });
    }

    /// Finish a login, restore the editor, and select the provider's default
    /// model, upstream's `completeProviderAuthentication`.
    pub(crate) async fn finish_login(
        &self,
        provider_name: &str,
        auth_type: AuthType,
        result: Result<(), String>,
    ) {
        *self.login_dialog.borrow_mut() = None;
        self.restore_editor();
        let action = if auth_type == AuthType::OAuth {
            format!("Logged in to {provider_name}")
        } else {
            format!("Saved API key for {provider_name}")
        };
        match result {
            Ok(()) => {
                let provider_id = self
                    .get_login_provider_options(None)
                    .into_iter()
                    .find(|provider| provider.name == provider_name)
                    .map(|provider| provider.id);
                if let Some(provider_id) = provider_id
                    && let Some(default_id) =
                        crate::model_resolver::default_model_per_provider(&provider_id)
                    && let Some(model) = self
                        .session()
                        .model_runtime()
                        .get_available_snapshot()
                        .into_iter()
                        .find(|model| model.provider.0 == provider_id && model.id == default_id)
                {
                    self.set_model(model, true).await;
                }
                self.update_available_provider_count().await;
                self.show_status(&format!(
                    "{action}. Credentials saved to {}",
                    crate::config::get_auth_path().display()
                ));
            }
            Err(error) => self.show_error(&format!("{action} failed: {error}")),
        }
    }

    /// Remove a stored credential.
    pub(crate) async fn logout_provider(&self, provider_id: &str) {
        match self
            .session()
            .model_runtime()
            .logout(provider_id, None)
            .await
        {
            Ok(()) => {
                self.update_available_provider_count().await;
                self.show_status(&format!("Removed stored credential for {provider_id}"));
            }
            Err(error) => self.show_error(&format!("Logout failed: {error}")),
        }
    }
}
