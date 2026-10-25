//! The platform-switched `tui.*` overrides and the `app.*` action table,
//! upstream's `KEYBINDINGS` in `src/core/keybindings.ts`.

use pi_tui::keybindings::{KeybindingDefinition, Keybindings};
use pi_tui::keys::KeyId;

use crate::config::{EnvLookup, default_env_lookup};

/// The host's `process.platform` spelling.
///
/// Node reports `darwin` and `win32` where [`std::env::consts::OS`] reports
/// `macos` and `windows`; every other value passes through unchanged.
#[must_use]
pub fn node_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        other => other,
    }
}

/// Whether the platform's defaults use the Windows key set, upstream's
/// `useWindowsKeybindings`.
///
/// Native Windows always does. Linux does inside WSL, detected by
/// `WSL_DISTRO_NAME` or `WSL_INTEROP`; an empty value counts as absent,
/// matching upstream's `Boolean(env.X)`.
#[must_use]
pub fn use_windows_keybindings(platform: &str, env: &EnvLookup) -> bool {
    platform == "win32"
        || (platform == "linux"
            && ["WSL_DISTRO_NAME", "WSL_INTEROP"]
                .iter()
                .any(|name| env(name).is_some_and(|value| !value.is_empty())))
}

/// A table row's default keys, resolved against the detected platform.
enum DefaultKeys {
    /// One spelling on every platform.
    Fixed(&'static [&'static str]),
    /// Empty on native Windows, `ctrl+z` elsewhere.
    Suspend,
    /// `alt+p` on the Windows key set, `shift+ctrl+p` elsewhere.
    CycleBackward,
    /// `ctrl+q` on the Windows key set, `alt+enter` elsewhere.
    FollowUp,
    /// `alt+q` on the Windows key set, `alt+up` elsewhere.
    Dequeue,
    /// `alt+v` on the Windows key set, `ctrl+v` elsewhere.
    PasteImage,
    /// `alt+left` first on macOS, `ctrl+left` first elsewhere.
    FoldOrUp,
    /// `alt+right` first on macOS, `ctrl+right` first elsewhere.
    UnfoldOrDown,
}

impl DefaultKeys {
    /// The keys this row binds for the detected flags.
    fn resolve(&self, windows: bool, native_windows: bool, darwin: bool) -> Vec<&'static str> {
        match self {
            Self::Fixed(keys) => keys.to_vec(),
            Self::Suspend => {
                if native_windows {
                    vec![]
                } else {
                    vec!["ctrl+z"]
                }
            }
            Self::CycleBackward => {
                if windows {
                    vec!["alt+p"]
                } else {
                    vec!["shift+ctrl+p"]
                }
            }
            Self::FollowUp => {
                if windows {
                    vec!["ctrl+q"]
                } else {
                    vec!["alt+enter"]
                }
            }
            Self::Dequeue => {
                if windows {
                    vec!["alt+q"]
                } else {
                    vec!["alt+up"]
                }
            }
            Self::PasteImage => {
                if windows {
                    vec!["alt+v"]
                } else {
                    vec!["ctrl+v"]
                }
            }
            Self::FoldOrUp => {
                if darwin {
                    vec!["alt+left", "ctrl+left"]
                } else {
                    vec!["ctrl+left", "alt+left"]
                }
            }
            Self::UnfoldOrDown => {
                if darwin {
                    vec!["alt+right", "ctrl+right"]
                } else {
                    vec!["ctrl+right", "alt+right"]
                }
            }
        }
    }
}

/// The `app.*` actions in upstream's declaration order.
const APP_BINDINGS: &[(&str, DefaultKeys, &str)] = &[
    (
        "app.interrupt",
        DefaultKeys::Fixed(&["escape"]),
        "Cancel or abort",
    ),
    ("app.clear", DefaultKeys::Fixed(&["ctrl+c"]), "Clear editor"),
    (
        "app.exit",
        DefaultKeys::Fixed(&["ctrl+d"]),
        "Exit when editor is empty",
    ),
    ("app.suspend", DefaultKeys::Suspend, "Suspend to background"),
    (
        "app.thinking.cycle",
        DefaultKeys::Fixed(&["shift+tab"]),
        "Cycle thinking level",
    ),
    (
        "app.thinking.save",
        DefaultKeys::Fixed(&["ctrl+s"]),
        "Save thinking level",
    ),
    (
        "app.model.cycleForward",
        DefaultKeys::Fixed(&["ctrl+p"]),
        "Cycle to next model",
    ),
    (
        "app.model.cycleBackward",
        DefaultKeys::CycleBackward,
        "Cycle to previous model",
    ),
    (
        "app.model.select",
        DefaultKeys::Fixed(&["ctrl+l"]),
        "Open model selector",
    ),
    (
        "app.tools.expand",
        DefaultKeys::Fixed(&["ctrl+o"]),
        "Toggle tool output",
    ),
    (
        "app.thinking.toggle",
        DefaultKeys::Fixed(&["ctrl+t"]),
        "Toggle thinking blocks",
    ),
    (
        "app.session.toggleNamedFilter",
        DefaultKeys::Fixed(&["ctrl+n"]),
        "Toggle named session filter",
    ),
    (
        "app.editor.external",
        DefaultKeys::Fixed(&["ctrl+g"]),
        "Open external editor",
    ),
    (
        "app.message.copy",
        DefaultKeys::Fixed(&["ctrl+x"]),
        "Copy message to clipboard",
    ),
    (
        "app.message.followUp",
        DefaultKeys::FollowUp,
        "Queue follow-up message",
    ),
    (
        "app.message.dequeue",
        DefaultKeys::Dequeue,
        "Restore queued messages",
    ),
    (
        "app.clipboard.pasteImage",
        DefaultKeys::PasteImage,
        "Paste image from clipboard (text fallback)",
    ),
    (
        "app.session.new",
        DefaultKeys::Fixed(&[]),
        "Start a new session",
    ),
    (
        "app.session.tree",
        DefaultKeys::Fixed(&[]),
        "Open session tree",
    ),
    (
        "app.session.fork",
        DefaultKeys::Fixed(&[]),
        "Fork current session",
    ),
    (
        "app.session.resume",
        DefaultKeys::Fixed(&[]),
        "Resume a session",
    ),
    (
        "app.tree.foldOrUp",
        DefaultKeys::FoldOrUp,
        "Fold tree branch or move up",
    ),
    (
        "app.tree.unfoldOrDown",
        DefaultKeys::UnfoldOrDown,
        "Unfold tree branch or move down",
    ),
    (
        "app.tree.editLabel",
        DefaultKeys::Fixed(&["shift+l"]),
        "Edit tree label",
    ),
    (
        "app.tree.toggleLabelTimestamp",
        DefaultKeys::Fixed(&["shift+t"]),
        "Toggle tree label timestamps",
    ),
    (
        "app.session.togglePath",
        DefaultKeys::Fixed(&["ctrl+p"]),
        "Toggle session path display",
    ),
    (
        "app.session.toggleSort",
        DefaultKeys::Fixed(&["ctrl+s"]),
        "Toggle session sort mode",
    ),
    (
        "app.session.rename",
        DefaultKeys::Fixed(&["ctrl+r"]),
        "Rename session",
    ),
    (
        "app.session.delete",
        DefaultKeys::Fixed(&["ctrl+d"]),
        "Delete session",
    ),
    (
        "app.session.deleteNoninvasive",
        DefaultKeys::Fixed(&["ctrl+backspace"]),
        "Delete session when query is empty",
    ),
    (
        "app.models.save",
        DefaultKeys::Fixed(&["ctrl+s"]),
        "Save model selection",
    ),
    (
        "app.models.enableAll",
        DefaultKeys::Fixed(&["ctrl+a"]),
        "Enable all models",
    ),
    (
        "app.models.clearAll",
        DefaultKeys::Fixed(&["ctrl+x"]),
        "Clear all models",
    ),
    (
        "app.models.toggleProvider",
        DefaultKeys::Fixed(&["ctrl+p"]),
        "Toggle all models for provider",
    ),
    (
        "app.models.reorderUp",
        DefaultKeys::Fixed(&["alt+up"]),
        "Move model up in order",
    ),
    (
        "app.models.reorderDown",
        DefaultKeys::Fixed(&["alt+down"]),
        "Move model down in order",
    ),
    (
        "app.tree.filter.default",
        DefaultKeys::Fixed(&["ctrl+d"]),
        "Tree filter: default view",
    ),
    (
        "app.tree.filter.noTools",
        DefaultKeys::Fixed(&["ctrl+t"]),
        "Tree filter: hide tool results",
    ),
    (
        "app.tree.filter.userOnly",
        DefaultKeys::Fixed(&["ctrl+u"]),
        "Tree filter: user messages only",
    ),
    (
        "app.tree.filter.labeledOnly",
        DefaultKeys::Fixed(&["ctrl+l"]),
        "Tree filter: labeled entries only",
    ),
    (
        "app.tree.filter.all",
        DefaultKeys::Fixed(&["ctrl+a"]),
        "Tree filter: show all entries",
    ),
    (
        "app.tree.filter.cycleForward",
        DefaultKeys::Fixed(&["ctrl+o"]),
        "Tree filter: cycle forward",
    ),
    (
        "app.tree.filter.cycleBackward",
        DefaultKeys::Fixed(&["shift+ctrl+o"]),
        "Tree filter: cycle backward",
    ),
];

/// The process keybinding registry: pi-tui's defaults with the four
/// platform-switched `tui.*` overrides and every `app.*` action.
#[must_use]
pub fn keybindings() -> Keybindings {
    keybindings_with(node_platform(), &default_env_lookup())
}

/// [`keybindings`] over an injected platform spelling and environment
/// lookup, upstream's module-load `useWindowsKeybindings()` read.
///
/// A re-registered `tui.*` action keeps its original position, mirroring
/// upstream's object-spread override.
#[must_use]
pub fn keybindings_with(platform: &str, env: &EnvLookup) -> Keybindings {
    let windows = use_windows_keybindings(platform, env);
    let native_windows = platform == "win32";
    let darwin = platform == "darwin";

    let mut registry = Keybindings::tui_defaults();
    let undo = if native_windows {
        vec!["ctrl+z"]
    } else if windows {
        vec!["alt+z"]
    } else {
        vec!["ctrl+-"]
    };
    override_defaults(&mut registry, "tui.editor.undo", undo);
    override_defaults(
        &mut registry,
        "tui.altScreen.previousPrompt",
        if windows {
            vec!["ctrl+up"]
        } else {
            vec!["ctrl+shift+up", "ctrl+up"]
        },
    );
    override_defaults(
        &mut registry,
        "tui.altScreen.nextPrompt",
        if windows {
            vec!["ctrl+down"]
        } else {
            vec!["ctrl+shift+down", "ctrl+down"]
        },
    );
    override_defaults(
        &mut registry,
        "tui.altScreen.search",
        if windows {
            vec!["ctrl+f"]
        } else {
            vec!["ctrl+shift+f"]
        },
    );

    for (action, keys, description) in APP_BINDINGS {
        let keys = keys.resolve(windows, native_windows, darwin);
        registry.register(*action, KeybindingDefinition::new(keys, Some(*description)));
    }

    registry
}

/// Replace one registry action's default keys, keeping its description and
/// position, upstream's spread override.
fn override_defaults(
    registry: &mut Keybindings,
    action: &str,
    keys: impl IntoIterator<Item = impl Into<KeyId>>,
) {
    let description = registry
        .definition(action)
        .and_then(|definition| definition.description);
    registry.register(action, KeybindingDefinition::new(keys, description));
}
