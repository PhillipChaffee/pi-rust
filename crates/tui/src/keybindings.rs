//! Global keybinding registry, ported from `packages/tui/src/keybindings.ts`
//! in earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#40).
//!
//! Upstream named action ids through declaration merging on the `Keybindings`
//! interface and passed definitions as plain records. The port restates that
//! as a runtime registry ([`Keybindings`]) keyed by string action ids:
//! [`Keybindings::tui_defaults`] seeds [`TUI_KEYBINDINGS`], and downstream
//! packages register additional `app.*` actions with [`Keybindings::register`]
//! before handing the registry to a [`KeybindingsManager`].
//!
//! [`KeybindingsManager`] resolves user bindings over the defaults, reports
//! direct user-binding conflicts without evicting defaults, and matches raw
//! terminal input against an action's keys through a [`KeyParser`] supplied by
//! the caller.

use std::sync::PoisonError;
use std::sync::{LazyLock, Mutex, MutexGuard};

use crate::keys::{KeyId, KeyParser};

/// One registered keybinding action: the keys bound when the user has not
/// overridden it, and the description rendered by config tooling.
#[derive(Debug, Clone)]
pub struct KeybindingDefinition {
    /// Keys bound when the user has not overridden the action; an empty list
    /// leaves the action unbound by default.
    pub default_keys: Vec<KeyId>,
    /// Human-readable description, upstream's optional `description`.
    pub description: Option<&'static str>,
}

impl KeybindingDefinition {
    /// Builds a definition from default keys and an optional description.
    #[must_use]
    pub fn new(
        default_keys: impl IntoIterator<Item = impl Into<KeyId>>,
        description: Option<&'static str>,
    ) -> Self {
        Self {
            default_keys: default_keys.into_iter().map(Into::into).collect(),
            description,
        }
    }
}

/// One [`TUI_KEYBINDINGS`] row: the upstream default for a `tui.*` action.
#[derive(Debug)]
pub struct TuiKeybinding {
    /// The action id, e.g. `"tui.editor.cursorUp"`.
    pub id: &'static str,
    /// Upstream `defaultKeys` — a single key or a list, flattened here.
    pub default_keys: &'static [&'static str],
    /// Upstream `description`.
    pub description: &'static str,
}

/// Upstream's `TUI_KEYBINDINGS` defaults, in upstream's declaration order.
pub const TUI_KEYBINDINGS: &[TuiKeybinding] = &[
    // Editor navigation and editing
    TuiKeybinding {
        id: "tui.editor.cursorUp",
        default_keys: &["up"],
        description: "Move cursor up",
    },
    TuiKeybinding {
        id: "tui.editor.cursorDown",
        default_keys: &["down"],
        description: "Move cursor down",
    },
    TuiKeybinding {
        id: "tui.editor.historyPrevious",
        default_keys: &[],
        description: "Select previous prompt history entry",
    },
    TuiKeybinding {
        id: "tui.editor.historyNext",
        default_keys: &[],
        description: "Select next prompt history entry",
    },
    TuiKeybinding {
        id: "tui.editor.cursorLeft",
        default_keys: &["left", "ctrl+b"],
        description: "Move cursor left",
    },
    TuiKeybinding {
        id: "tui.editor.cursorRight",
        default_keys: &["right", "ctrl+f"],
        description: "Move cursor right",
    },
    TuiKeybinding {
        id: "tui.editor.cursorWordLeft",
        default_keys: &["alt+left", "ctrl+left", "alt+b"],
        description: "Move cursor word left",
    },
    TuiKeybinding {
        id: "tui.editor.cursorWordRight",
        default_keys: &["alt+right", "ctrl+right", "alt+f"],
        description: "Move cursor word right",
    },
    TuiKeybinding {
        id: "tui.editor.cursorLineStart",
        default_keys: &["home", "ctrl+home", "ctrl+a"],
        description: "Move to line start",
    },
    TuiKeybinding {
        id: "tui.editor.cursorLineEnd",
        default_keys: &["end", "ctrl+end", "ctrl+e"],
        description: "Move to line end",
    },
    TuiKeybinding {
        id: "tui.editor.jumpForward",
        default_keys: &["ctrl+]"],
        description: "Jump forward to character",
    },
    TuiKeybinding {
        id: "tui.editor.jumpBackward",
        default_keys: &["ctrl+alt+]"],
        description: "Jump backward to character",
    },
    TuiKeybinding {
        id: "tui.editor.pageUp",
        default_keys: &["pageUp", "ctrl+pageUp"],
        description: "Page up",
    },
    TuiKeybinding {
        id: "tui.editor.pageDown",
        default_keys: &["pageDown", "ctrl+pageDown"],
        description: "Page down",
    },
    TuiKeybinding {
        id: "tui.editor.deleteCharBackward",
        default_keys: &["backspace"],
        description: "Delete character backward",
    },
    TuiKeybinding {
        id: "tui.editor.deleteCharForward",
        default_keys: &["delete", "ctrl+d"],
        description: "Delete character forward",
    },
    TuiKeybinding {
        id: "tui.editor.deleteWordBackward",
        default_keys: &["ctrl+w", "alt+backspace"],
        description: "Delete word backward",
    },
    TuiKeybinding {
        id: "tui.editor.deleteWordForward",
        default_keys: &["alt+d", "alt+delete"],
        description: "Delete word forward",
    },
    TuiKeybinding {
        id: "tui.editor.deleteToLineStart",
        default_keys: &["ctrl+u"],
        description: "Delete to line start",
    },
    TuiKeybinding {
        id: "tui.editor.deleteToLineEnd",
        default_keys: &["ctrl+k"],
        description: "Delete to line end",
    },
    TuiKeybinding {
        id: "tui.editor.yank",
        default_keys: &["ctrl+y"],
        description: "Yank",
    },
    TuiKeybinding {
        id: "tui.editor.yankPop",
        default_keys: &["alt+y"],
        description: "Yank pop",
    },
    TuiKeybinding {
        id: "tui.editor.undo",
        default_keys: &["ctrl+-"],
        description: "Undo",
    },
    TuiKeybinding {
        id: "tui.input.newLine",
        default_keys: &["shift+enter", "ctrl+j"],
        description: "Insert newline",
    },
    TuiKeybinding {
        id: "tui.input.submit",
        default_keys: &["enter"],
        description: "Submit input",
    },
    TuiKeybinding {
        id: "tui.input.tab",
        default_keys: &["tab"],
        description: "Tab / autocomplete",
    },
    TuiKeybinding {
        id: "tui.input.copy",
        default_keys: &["ctrl+c"],
        description: "Copy selection",
    },
    TuiKeybinding {
        id: "tui.select.up",
        default_keys: &["up"],
        description: "Move selection up",
    },
    TuiKeybinding {
        id: "tui.select.down",
        default_keys: &["down"],
        description: "Move selection down",
    },
    TuiKeybinding {
        id: "tui.select.pageUp",
        default_keys: &["pageUp"],
        description: "Selection page up",
    },
    TuiKeybinding {
        id: "tui.select.pageDown",
        default_keys: &["pageDown"],
        description: "Selection page down",
    },
    TuiKeybinding {
        id: "tui.select.confirm",
        default_keys: &["enter"],
        description: "Confirm selection",
    },
    TuiKeybinding {
        id: "tui.select.cancel",
        default_keys: &["escape", "ctrl+c"],
        description: "Cancel selection",
    },
    // These intentionally shadow the unmodified editor bindings in fullscreen mode.
    TuiKeybinding {
        id: "tui.altScreen.pageUp",
        default_keys: &["pageUp"],
        description: "Scroll viewport up one page",
    },
    TuiKeybinding {
        id: "tui.altScreen.pageDown",
        default_keys: &["pageDown"],
        description: "Scroll viewport down one page",
    },
    TuiKeybinding {
        id: "tui.altScreen.halfPageUp",
        default_keys: &[],
        description: "Scroll viewport up half a page",
    },
    TuiKeybinding {
        id: "tui.altScreen.halfPageDown",
        default_keys: &[],
        description: "Scroll viewport down half a page",
    },
    TuiKeybinding {
        id: "tui.altScreen.lineUp",
        default_keys: &[],
        description: "Scroll viewport up one line",
    },
    TuiKeybinding {
        id: "tui.altScreen.lineDown",
        default_keys: &[],
        description: "Scroll viewport down one line",
    },
    TuiKeybinding {
        id: "tui.altScreen.previousPrompt",
        default_keys: &["ctrl+shift+up", "ctrl+up"],
        description: "Jump to previous semantic prompt",
    },
    TuiKeybinding {
        id: "tui.altScreen.nextPrompt",
        default_keys: &["ctrl+shift+down", "ctrl+down"],
        description: "Jump to next semantic prompt",
    },
    TuiKeybinding {
        id: "tui.altScreen.search",
        default_keys: &["ctrl+shift+f"],
        description: "Search the primary scroll view",
    },
    TuiKeybinding {
        id: "tui.altScreen.searchNext",
        default_keys: &["enter", "ctrl+g"],
        description: "Select the next search match",
    },
    TuiKeybinding {
        id: "tui.altScreen.searchPrevious",
        default_keys: &["shift+enter", "ctrl+shift+g"],
        description: "Select the previous search match",
    },
    TuiKeybinding {
        id: "tui.altScreen.searchClose",
        default_keys: &["escape"],
        description: "Close transcript search",
    },
    TuiKeybinding {
        id: "tui.altScreen.top",
        default_keys: &["home"],
        description: "Scroll viewport to top",
    },
    TuiKeybinding {
        id: "tui.altScreen.bottom",
        default_keys: &["end"],
        description: "Scroll viewport to bottom",
    },
];

/// Runtime registry of keybinding definitions keyed by action id.
///
/// This restates upstream's declaration-merged `Keybindings` interface plus
/// its `KeybindingDefinitions` record: the action ids upstream froze in the
/// type system live here as strings, and a downstream package adds its own
/// actions by registering entries on a seeded registry instead of merging an
/// interface. Registration preserves first-declaration order; re-registering
/// an id replaces its definition in place.
#[derive(Debug, Clone, Default)]
pub struct Keybindings {
    definitions: Vec<(String, KeybindingDefinition)>,
}

impl Keybindings {
    /// Upstream's `TUI_KEYBINDINGS` as a registry.
    #[must_use]
    pub fn tui_defaults() -> Self {
        let mut registry = Self {
            definitions: Vec::new(),
        };
        for entry in TUI_KEYBINDINGS {
            registry.register(
                entry.id,
                KeybindingDefinition::new(
                    entry.default_keys.iter().copied(),
                    Some(entry.description),
                ),
            );
        }
        registry
    }

    /// Registers or replaces one action definition. Replacing keeps the id's
    /// original position, mirroring object-literal spread semantics.
    pub fn register(
        &mut self,
        action: impl Into<String>,
        definition: KeybindingDefinition,
    ) -> &mut Self {
        let action = action.into();
        match self.definitions.iter_mut().find(|(id, _)| *id == action) {
            Some((_, existing)) => *existing = definition,
            None => self.definitions.push((action, definition)),
        }
        self
    }

    /// The definition for `action`, or `None` when the action is not registered.
    #[must_use]
    pub fn definition(&self, action: &str) -> Option<&KeybindingDefinition> {
        self.definitions
            .iter()
            .find(|(id, _)| id == action)
            .map(|(_, definition)| definition)
    }

    /// The registered action ids in declaration order.
    pub fn action_ids(&self) -> impl Iterator<Item = &str> {
        self.definitions.iter().map(|(id, _)| id.as_str())
    }
}

/// A key claimed by more than one user binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeybindingConflict {
    /// The contested key identifier.
    pub key: KeyId,
    /// The actions claiming it, in first-claim order.
    pub keybindings: Vec<String>,
}

/// User-supplied overrides, restated from upstream's `KeybindingsConfig`
/// record with insertion order preserved.
///
/// A user config entry's position decides the order of
/// [`KeybindingsManager::get_conflicts`] output. Binding an action to an
/// empty list unbinds it; leaving it absent keeps the definition's defaults.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KeybindingsConfig {
    bindings: Vec<(String, Vec<KeyId>)>,
}

impl KeybindingsConfig {
    /// An empty configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Binds `action` to `keys`, replacing any earlier entry for the action in
    /// place.
    #[must_use]
    pub fn bind(
        mut self,
        action: impl Into<String>,
        keys: impl IntoIterator<Item = impl Into<KeyId>>,
    ) -> Self {
        let action = action.into();
        let keys: Vec<KeyId> = keys.into_iter().map(Into::into).collect();
        match self.bindings.iter_mut().find(|(bound, _)| *bound == action) {
            Some((_, existing)) => *existing = keys,
            None => self.bindings.push((action, keys)),
        }
        self
    }

    /// The keys bound to `action`, or `None` when absent.
    #[must_use]
    pub fn get(&self, action: &str) -> Option<&[KeyId]> {
        self.bindings
            .iter()
            .find(|(bound, _)| bound == action)
            .map(|(_, keys)| keys.as_slice())
    }

    /// The entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[KeyId])> {
        self.bindings
            .iter()
            .map(|(action, keys)| (action.as_str(), keys.as_slice()))
    }

    /// How many actions this configuration binds.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether this configuration binds nothing.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// Resolves keybinding definitions against user bindings and matches terminal
/// input against the resolved keys.
///
/// Upstream keyed actions by declaration-merged id literals; here every
/// lookup takes the action id string, and unknown ids simply match nothing or
/// resolve to empty lists.
#[derive(Debug)]
pub struct KeybindingsManager {
    definitions: Keybindings,
    user_bindings: KeybindingsConfig,
    keys_by_id: std::collections::HashMap<String, Vec<KeyId>>,
    conflicts: Vec<KeybindingConflict>,
}

impl KeybindingsManager {
    /// A manager for `definitions` with no user overrides.
    #[must_use]
    pub fn new(definitions: Keybindings) -> Self {
        Self::with_user_bindings(definitions, KeybindingsConfig::new())
    }

    /// A manager for `definitions` with `user_bindings` layered over the
    /// defaults. Entries whose action is not in the registry are ignored.
    #[must_use]
    pub fn with_user_bindings(definitions: Keybindings, user_bindings: KeybindingsConfig) -> Self {
        let mut manager = Self {
            definitions,
            user_bindings,
            keys_by_id: std::collections::HashMap::new(),
            conflicts: Vec::new(),
        };
        manager.rebuild();
        manager
    }

    fn rebuild(&mut self) {
        let mut keys_by_id = std::collections::HashMap::new();
        let mut conflicts = Vec::new();

        // Direct user-claims conflicts: a key claimed by two or more user
        // bindings, in the order the claims were inserted.
        let mut user_claims: Vec<(KeyId, Vec<String>)> = Vec::new();
        for (action, keys) in self.user_bindings.iter() {
            if self.definitions.definition(action).is_none() {
                continue;
            }
            for key in normalize_keys(keys) {
                if let Some((_, claimants)) =
                    user_claims.iter_mut().find(|(claimed, _)| *claimed == key)
                {
                    claimants.push(action.to_string());
                } else {
                    user_claims.push((key, vec![action.to_string()]));
                }
            }
        }
        conflicts.extend(
            user_claims
                .into_iter()
                .filter(|(_, claimants)| claimants.len() > 1)
                .map(|(key, claimants)| KeybindingConflict {
                    key,
                    keybindings: claimants,
                }),
        );

        for id in self.definitions.action_ids() {
            let keys = match self.user_bindings.get(id) {
                Some(user_keys) => normalize_keys(user_keys),
                None => self
                    .definitions
                    .definition(id)
                    .map(|definition| normalize_keys(&definition.default_keys))
                    .unwrap_or_default(),
            };
            keys_by_id.insert(id.to_string(), keys);
        }

        self.keys_by_id = keys_by_id;
        self.conflicts = conflicts;
    }

    /// Whether the input matches any key bound to `action`. `parser` carries
    /// the Kitty protocol state; see the crate docs.
    #[must_use]
    pub fn matches(&self, parser: &KeyParser, data: &str, action: &str) -> bool {
        self.keys_by_id
            .get(action)
            .is_some_and(|keys| keys.iter().any(|key| parser.matches_key(data, key)))
    }

    /// The keys bound to `action`, in order; empty when the action is unbound
    /// or unknown.
    #[must_use]
    pub fn get_keys(&self, action: &str) -> Vec<KeyId> {
        self.keys_by_id.get(action).cloned().unwrap_or_default()
    }

    /// The registered definition for `action`.
    #[must_use]
    pub fn get_definition(&self, action: &str) -> Option<&KeybindingDefinition> {
        self.definitions.definition(action)
    }

    /// The direct user-binding conflicts, in first-claim order.
    #[must_use]
    pub fn get_conflicts(&self) -> Vec<KeybindingConflict> {
        self.conflicts.clone()
    }

    /// Replaces the user overrides and re-resolves every action.
    pub fn set_user_bindings(&mut self, user_bindings: KeybindingsConfig) {
        self.user_bindings = user_bindings;
        self.rebuild();
    }

    /// A copy of the current user overrides.
    #[must_use]
    pub fn get_user_bindings(&self) -> KeybindingsConfig {
        self.user_bindings.clone()
    }

    /// Every registered action resolved to its effective keys. Upstream
    /// collapses a single resolved key to a scalar for the keybindings.json
    /// shape; the port keeps each action's full list, and the config
    /// serializer owns the JSON shape.
    #[must_use]
    pub fn get_resolved_bindings(&self) -> KeybindingsConfig {
        let mut resolved = KeybindingsConfig::new();
        for id in self.definitions.action_ids() {
            let keys = self.keys_by_id.get(id).cloned().unwrap_or_default();
            resolved = resolved.bind(id, keys);
        }
        resolved
    }
}

/// Deduplicates keys preserving first occurrence, restating upstream's
/// `normalizeKeys`.
fn normalize_keys(keys: &[KeyId]) -> Vec<KeyId> {
    let mut result: Vec<KeyId> = Vec::new();
    for key in keys {
        if !result.contains(key) {
            result.push(key.clone());
        }
    }
    result
}

static GLOBAL_KEYBINDINGS: LazyLock<Mutex<KeybindingsManager>> =
    LazyLock::new(|| Mutex::new(KeybindingsManager::new(Keybindings::tui_defaults())));

fn lock_global() -> MutexGuard<'static, KeybindingsManager> {
    GLOBAL_KEYBINDINGS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Installs the process-wide keybindings manager, restating upstream's
/// `setKeybindings`.
pub fn set_keybindings(keybindings: KeybindingsManager) {
    *lock_global() = keybindings;
}

/// The process-wide manager, restating upstream's `getKeybindings`: the
/// installed manager, or a lazily created one over [`TUI_KEYBINDINGS`].
pub fn get_keybindings() -> MutexGuard<'static, KeybindingsManager> {
    lock_global()
}
