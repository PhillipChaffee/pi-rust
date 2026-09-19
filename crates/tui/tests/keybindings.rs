//! Port of `packages/tui/test/keybindings.test.ts` — 1:1 against upstream pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (#40) — plus boundary tests for
//! the manager surfaces upstream leaves untested (the global accessor pair,
//! user-binding replacement, resolved bindings, and registry registration)
//! so the coverage gate binds.

use pi_tui::keybindings::{
    KeybindingDefinition, Keybindings, KeybindingsConfig, KeybindingsManager, TUI_KEYBINDINGS,
    get_keybindings, set_keybindings,
};
use pi_tui::keys::KeyParser;

fn manager() -> KeybindingsManager {
    KeybindingsManager::new(Keybindings::tui_defaults())
}

#[test]
fn binds_ctrl_j_as_a_default_newline_alias() {
    let keybindings = manager();

    assert_eq!(
        keybindings.get_keys("tui.input.newLine"),
        vec!["shift+enter", "ctrl+j"]
    );
    assert!(keybindings.matches(&KeyParser::new(), "\n", "tui.input.newLine"));
    assert!(keybindings.matches(&KeyParser::new(), "\x1b[106;5u", "tui.input.newLine"));
}

#[test]
fn binds_modified_and_unmodified_editor_viewport_navigation() {
    let keybindings = manager();

    assert_eq!(
        keybindings.get_keys("tui.editor.cursorLineStart"),
        vec!["home", "ctrl+home", "ctrl+a"]
    );
    assert_eq!(
        keybindings.get_keys("tui.editor.cursorLineEnd"),
        vec!["end", "ctrl+end", "ctrl+e"]
    );
    assert_eq!(
        keybindings.get_keys("tui.editor.pageUp"),
        vec!["pageUp", "ctrl+pageUp"]
    );
    assert_eq!(
        keybindings.get_keys("tui.editor.pageDown"),
        vec!["pageDown", "ctrl+pageDown"]
    );
}

#[test]
fn leaves_dedicated_prompt_history_navigation_unbound_by_default() {
    let keybindings = manager();

    assert_eq!(
        keybindings.get_keys("tui.editor.historyPrevious"),
        Vec::<String>::new()
    );
    assert_eq!(
        keybindings.get_keys("tui.editor.historyNext"),
        Vec::<String>::new()
    );
}

#[test]
fn binds_unmodified_terminal_viewport_shortcuts_to_alternate_screen_navigation() {
    let keybindings = manager();

    assert_eq!(keybindings.get_keys("tui.altScreen.pageUp"), vec!["pageUp"]);
    assert_eq!(
        keybindings.get_keys("tui.altScreen.pageDown"),
        vec!["pageDown"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.halfPageUp"),
        Vec::<String>::new()
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.halfPageDown"),
        Vec::<String>::new()
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.lineUp"),
        Vec::<String>::new()
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.lineDown"),
        Vec::<String>::new()
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.previousPrompt"),
        vec!["ctrl+shift+up", "ctrl+up"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.nextPrompt"),
        vec!["ctrl+shift+down", "ctrl+down"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.search"),
        vec!["ctrl+shift+f"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.searchNext"),
        vec!["enter", "ctrl+g"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.searchPrevious"),
        vec!["shift+enter", "ctrl+shift+g"]
    );
    assert_eq!(
        keybindings.get_keys("tui.altScreen.searchClose"),
        vec!["escape"]
    );
    assert_eq!(keybindings.get_keys("tui.altScreen.top"), vec!["home"]);
    assert_eq!(keybindings.get_keys("tui.altScreen.bottom"), vec!["end"]);
}

#[test]
fn does_not_evict_selector_confirm_when_input_submit_is_rebound() {
    let keybindings = KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new().bind("tui.input.submit", ["enter", "ctrl+enter"]),
    );

    assert_eq!(
        keybindings.get_keys("tui.input.submit"),
        vec!["enter", "ctrl+enter"]
    );
    assert_eq!(keybindings.get_keys("tui.select.confirm"), vec!["enter"]);
}

#[test]
fn does_not_evict_cursor_bindings_when_another_action_reuses_the_same_key() {
    let keybindings = KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new().bind("tui.select.up", ["up", "ctrl+p"]),
    );

    assert_eq!(keybindings.get_keys("tui.select.up"), vec!["up", "ctrl+p"]);
    assert_eq!(keybindings.get_keys("tui.editor.cursorUp"), vec!["up"]);
}

#[test]
fn still_reports_direct_user_binding_conflicts_without_evicting_defaults() {
    let keybindings = KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new()
            .bind("tui.input.submit", ["ctrl+x"])
            .bind("tui.select.confirm", ["ctrl+x"]),
    );

    assert_eq!(
        keybindings.get_conflicts(),
        vec![pi_tui::keybindings::KeybindingConflict {
            key: "ctrl+x".to_string(),
            keybindings: vec![
                "tui.input.submit".to_string(),
                "tui.select.confirm".to_string()
            ],
        }]
    );
    assert_eq!(
        keybindings.get_keys("tui.editor.cursorLeft"),
        vec!["left", "ctrl+b"]
    );
}

// =============================================================================
// Boundary tests (additions beyond upstream)
// =============================================================================

#[test]
fn tui_defaults_table_carries_every_registered_action_with_descriptions() {
    let registry = Keybindings::tui_defaults();

    for entry in TUI_KEYBINDINGS {
        assert_eq!(
            registry
                .definition(entry.id)
                .map(|definition| (definition.default_keys.clone(), definition.description)),
            Some((
                entry
                    .default_keys
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                Some(entry.description)
            )),
            "default mismatch for {}",
            entry.id
        );
    }
}

#[test]
fn registry_registration_appends_and_replaces_in_place() {
    let mut registry = Keybindings::tui_defaults();
    registry.register(
        "app.interrupt",
        KeybindingDefinition::new(["escape"], Some("Cancel or abort")),
    );
    registry.register(
        "tui.input.submit",
        KeybindingDefinition::new(["ctrl+x"], Some("Submit input")),
    );

    let ids: Vec<&str> = registry.action_ids().collect();
    let submit = ids.iter().position(|id| *id == "tui.input.submit");
    let confirm = ids.iter().position(|id| *id == "tui.select.confirm");
    let interrupt = ids.iter().position(|id| *id == "app.interrupt");
    // Replacement keeps the id's original position; new actions append.
    assert!(submit < confirm);
    assert!(interrupt > confirm);
    assert_eq!(
        registry
            .definition("tui.input.submit")
            .map(|definition| definition.default_keys.clone()),
        Some(vec!["ctrl+x".to_string()])
    );
}

#[test]
fn set_user_bindings_resolves_and_reports() {
    let mut keybindings = manager();
    let overrides = KeybindingsConfig::new().bind("tui.input.submit", ["enter", "ctrl+x"]);

    keybindings.set_user_bindings(overrides.clone());
    assert_eq!(keybindings.get_user_bindings(), overrides);
    assert_eq!(
        keybindings.get_keys("tui.input.submit"),
        vec!["enter", "ctrl+x"]
    );
    // Explicit empty list unbinds; absent ids fall back to defaults.
    keybindings
        .set_user_bindings(KeybindingsConfig::new().bind("tui.input.submit", Vec::<String>::new()));
    assert_eq!(
        keybindings.get_keys("tui.input.submit"),
        Vec::<String>::new()
    );
    assert_eq!(keybindings.get_keys("tui.editor.cursorUp"), vec!["up"]);
    // Reverting to an empty config restores defaults.
    keybindings.set_user_bindings(KeybindingsConfig::new());
    assert_eq!(keybindings.get_keys("tui.input.submit"), vec!["enter"]);
}

#[test]
fn resolved_bindings_cover_every_registered_action_in_order() {
    let keybindings = manager();
    let resolved = keybindings.get_resolved_bindings();

    assert_eq!(resolved.len(), TUI_KEYBINDINGS.len());
    assert_eq!(
        resolved.get("tui.input.submit").map(<[String]>::to_vec),
        Some(vec!["enter".to_string()])
    );
    // The first declared action resolves first.
    assert_eq!(
        resolved.iter().next().map(|(action, _)| action),
        Some("tui.editor.cursorUp")
    );
    assert!(!resolved.is_empty());
    assert_eq!(resolved.get("tui.does.notExist"), None);
}

#[test]
fn unknown_actions_match_nothing_and_resolve_to_empty() {
    let keybindings = manager();

    assert!(keybindings.get_keys("tui.does.notExist").is_empty());
    assert!(keybindings.get_definition("tui.does.notExist").is_none());
    assert!(!keybindings.matches(&KeyParser::new(), "\r", "tui.does.notExist"));
    // User overrides for unregistered actions are ignored.
    let keybindings = KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new().bind("tui.does.notExist", ["ctrl+x"]),
    );
    assert!(keybindings.get_conflicts().is_empty());
}

#[test]
fn user_binding_definitions_ignore_entries_without_a_registered_action() {
    let keybindings = KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new().bind("app.unknown", ["ctrl+x"]),
    );
    assert!(keybindings.get_conflicts().is_empty());
    assert_eq!(keybindings.get_keys("tui.input.submit"), vec!["enter"]);
}

#[test]
fn config_len_and_empty_reflect_bound_actions() {
    let empty = KeybindingsConfig::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);

    let bound = KeybindingsConfig::new().bind("tui.input.submit", ["enter"]);
    assert!(!bound.is_empty());
    assert_eq!(bound.len(), 1);
    // Rebinding the same action replaces it in place.
    let rebound = bound.bind("tui.input.submit", ["ctrl+x"]);
    assert_eq!(rebound.len(), 1);
    assert_eq!(
        rebound.get("tui.input.submit").map(<[String]>::to_vec),
        Some(vec!["ctrl+x".to_string()])
    );
}

#[test]
fn global_accessor_installs_and_returns_the_manager() {
    set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
    assert_eq!(
        get_keybindings().get_keys("tui.input.submit"),
        vec!["enter"]
    );
    assert!(
        get_keybindings()
            .get_definition("tui.editor.cursorUp")
            .is_some_and(|definition| definition.description == Some("Move cursor up"))
    );

    set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
    assert_eq!(
        get_keybindings().get_keys("tui.input.submit"),
        vec!["enter"]
    );
}
