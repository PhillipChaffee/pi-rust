//! Legacy-name migration, config projection, and the app keybinding manager,
//! upstream's `KeybindingsManager` and its helpers in
//! `src/core/keybindings.ts`.

use std::path::{Path, PathBuf};

use pi_tui::keybindings::{KeybindingsConfig, KeybindingsManager};
use pi_tui::keys::KeyId;
use serde_json::{Map, Value};

use crate::utils::text::strip_bom;

use super::definition::keybindings;

/// The legacy action names and their namespaced replacements, upstream's
/// `KEYBINDING_NAME_MIGRATIONS`.
const KEYBINDING_NAME_MIGRATIONS: &[(&str, &str)] = &[
    ("cursorUp", "tui.editor.cursorUp"),
    ("cursorDown", "tui.editor.cursorDown"),
    ("cursorLeft", "tui.editor.cursorLeft"),
    ("cursorRight", "tui.editor.cursorRight"),
    ("cursorWordLeft", "tui.editor.cursorWordLeft"),
    ("cursorWordRight", "tui.editor.cursorWordRight"),
    ("cursorLineStart", "tui.editor.cursorLineStart"),
    ("cursorLineEnd", "tui.editor.cursorLineEnd"),
    ("jumpForward", "tui.editor.jumpForward"),
    ("jumpBackward", "tui.editor.jumpBackward"),
    ("pageUp", "tui.editor.pageUp"),
    ("pageDown", "tui.editor.pageDown"),
    ("deleteCharBackward", "tui.editor.deleteCharBackward"),
    ("deleteCharForward", "tui.editor.deleteCharForward"),
    ("deleteWordBackward", "tui.editor.deleteWordBackward"),
    ("deleteWordForward", "tui.editor.deleteWordForward"),
    ("deleteToLineStart", "tui.editor.deleteToLineStart"),
    ("deleteToLineEnd", "tui.editor.deleteToLineEnd"),
    ("yank", "tui.editor.yank"),
    ("yankPop", "tui.editor.yankPop"),
    ("undo", "tui.editor.undo"),
    ("newLine", "tui.input.newLine"),
    ("submit", "tui.input.submit"),
    ("tab", "tui.input.tab"),
    ("copy", "tui.input.copy"),
    ("selectUp", "tui.select.up"),
    ("selectDown", "tui.select.down"),
    ("selectPageUp", "tui.select.pageUp"),
    ("selectPageDown", "tui.select.pageDown"),
    ("selectConfirm", "tui.select.confirm"),
    ("selectCancel", "tui.select.cancel"),
    ("interrupt", "app.interrupt"),
    ("clear", "app.clear"),
    ("exit", "app.exit"),
    ("suspend", "app.suspend"),
    ("cycleThinkingLevel", "app.thinking.cycle"),
    ("cycleModelForward", "app.model.cycleForward"),
    ("cycleModelBackward", "app.model.cycleBackward"),
    ("selectModel", "app.model.select"),
    ("expandTools", "app.tools.expand"),
    ("toggleThinking", "app.thinking.toggle"),
    ("toggleSessionNamedFilter", "app.session.toggleNamedFilter"),
    ("externalEditor", "app.editor.external"),
    ("followUp", "app.message.followUp"),
    ("dequeue", "app.message.dequeue"),
    ("pasteImage", "app.clipboard.pasteImage"),
    ("newSession", "app.session.new"),
    ("tree", "app.session.tree"),
    ("fork", "app.session.fork"),
    ("resume", "app.session.resume"),
    ("treeFoldOrUp", "app.tree.foldOrUp"),
    ("treeUnfoldOrDown", "app.tree.unfoldOrDown"),
    ("treeEditLabel", "app.tree.editLabel"),
    ("treeToggleLabelTimestamp", "app.tree.toggleLabelTimestamp"),
    ("toggleSessionPath", "app.session.togglePath"),
    ("toggleSessionSort", "app.session.toggleSort"),
    ("renameSession", "app.session.rename"),
    ("deleteSession", "app.session.delete"),
    ("deleteSessionNoninvasive", "app.session.deleteNoninvasive"),
];

/// The namespaced replacement for a legacy action name.
fn namespaced_name(key: &str) -> Option<&'static str> {
    KEYBINDING_NAME_MIGRATIONS
        .iter()
        .find(|(legacy, _)| *legacy == key)
        .map(|(_, next)| *next)
}

/// Rewrite legacy action names and order the config by the registry's
/// declaration order, upstream's `migrateKeybindingsConfig`.
///
/// Returns the rewritten config and whether any key was renamed. When both a
/// legacy name and its replacement exist, the replacement wins and the legacy
/// entry is dropped. Unregistered keys keep their values, sorted after the
/// registry's actions.
#[must_use]
pub fn migrate_keybindings_config(config: &Map<String, Value>) -> (Map<String, Value>, bool) {
    let mut migrated = false;
    let mut renamed = Map::new();

    for (key, value) in config {
        let next_key = namespaced_name(key).unwrap_or(key.as_str());
        if next_key != key.as_str() {
            migrated = true;
            if config.contains_key(next_key) {
                continue;
            }
        }
        renamed.insert(next_key.to_string(), value.clone());
    }

    (order_keybindings_config(&renamed), migrated)
}

/// [`migrate_keybindings_config`] under the migrations sweep's seam, which
/// lets a migrator decline a parsed file. This one always answers.
#[must_use]
pub fn migrate_keybindings_config_value(
    config: &Map<String, Value>,
) -> Option<(Map<String, Value>, bool)> {
    Some(migrate_keybindings_config(config))
}

/// Order the config by the registry's declaration order, extras sorted,
/// upstream's `orderKeybindingsConfig`.
fn order_keybindings_config(config: &Map<String, Value>) -> Map<String, Value> {
    let registry = keybindings();
    let mut ordered = Map::new();
    for action in registry.action_ids() {
        if let Some(value) = config.get(action) {
            ordered.insert(action.to_string(), value.clone());
        }
    }

    let mut extras: Vec<&String> = config
        .keys()
        .filter(|key| !ordered.contains_key(*key))
        .collect();
    extras.sort();
    for key in extras {
        ordered.insert(key.clone(), config[key].clone());
    }
    ordered
}

/// Project a raw keybindings object onto user bindings, upstream's
/// `toKeybindingsConfig`: string values bind one key, string arrays bind
/// several, everything else is ignored.
#[must_use]
pub fn keybindings_config_from_value(config: &Map<String, Value>) -> KeybindingsConfig {
    let mut bindings = KeybindingsConfig::new();
    for (action, value) in config {
        let keys: Vec<KeyId> = match value {
            Value::String(key) => vec![key.clone()],
            Value::Array(entries) if entries.iter().all(Value::is_string) => entries
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect(),
            _ => continue,
        };
        bindings = bindings.bind(action.clone(), keys);
    }
    bindings
}

/// Read `keybindings.json`, upstream's `loadRawConfig`.
///
/// A missing, unreadable, malformed, or non-object file reads as no config.
/// The BOM strip mirrors Node's UTF-8 read over upstream's `stripBom`.
fn load_raw_config(path: &Path) -> Option<Map<String, Value>> {
    let content = std::fs::read_to_string(path).ok()?;
    let parsed: Value = serde_json::from_str(strip_bom(&content)).ok()?;
    parsed.as_object().cloned()
}

/// Load user bindings from `keybindings.json`, upstream's
/// `KeybindingsManager.loadFromFile`.
fn load_from_file(path: &Path) -> KeybindingsConfig {
    load_raw_config(path).map_or_else(KeybindingsConfig::new, |config| {
        keybindings_config_from_value(&migrate_keybindings_config(&config).0)
    })
}

/// The app keybinding manager, upstream's `KeybindingsManager`: pi-tui's
/// resolution machinery over `<agent dir>/keybindings.json`, with a reload
/// that re-reads the file.
#[derive(Debug)]
pub struct AppKeybindingsManager {
    manager: KeybindingsManager,
    config_path: PathBuf,
}

impl AppKeybindingsManager {
    /// A manager for `agent_dir`, loading `<agent_dir>/keybindings.json`
    /// when it exists.
    #[must_use]
    pub fn create(agent_dir: &Path) -> Self {
        let config_path = agent_dir.join("keybindings.json");
        let user_bindings = load_from_file(&config_path);
        Self {
            manager: KeybindingsManager::with_user_bindings(keybindings(), user_bindings),
            config_path,
        }
    }

    /// Re-read the config file and re-resolve every action.
    pub fn reload(&mut self) {
        self.manager
            .set_user_bindings(load_from_file(&self.config_path));
    }

    /// Every action resolved to its effective keys.
    #[must_use]
    pub fn get_effective_config(&self) -> KeybindingsConfig {
        self.manager.get_resolved_bindings()
    }

    /// The user overrides currently loaded.
    #[must_use]
    pub fn get_user_bindings(&self) -> KeybindingsConfig {
        self.manager.get_user_bindings()
    }

    /// The resolution machinery, for key matching in the interactive shell.
    #[must_use]
    pub const fn manager(&self) -> &KeybindingsManager {
        &self.manager
    }
}
