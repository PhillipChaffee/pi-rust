//! The editor history-keybindings suite, ported 1:1 from
//! `packages/tui/test/editor-history-keybindings.test.ts` in
//! earendil-works/pi at commit `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`
//! (#47).
//!
//! Restatements against upstream: the global `setKeybindings` restore in
//! `afterEach` becomes a guard that reinstalls the default manager when the
//! test ends.
#[path = "tui_support/mod.rs"]
mod tui_support;

use pi_tui::components::Editor;
use pi_tui::keybindings::{Keybindings, KeybindingsConfig, KeybindingsManager, set_keybindings};

/// Upstream's `afterEach` restore.
struct RestoreDefaults;

impl Drop for RestoreDefaults {
    fn drop(&mut self) {
        set_keybindings(KeybindingsManager::new(Keybindings::tui_defaults()));
    }
}

#[test]
fn browses_history_directly_without_first_moving_the_cursor() {
    let _restore = RestoreDefaults;
    set_keybindings(KeybindingsManager::with_user_bindings(
        Keybindings::tui_defaults(),
        KeybindingsConfig::new()
            .bind("tui.editor.historyPrevious", ["ctrl+p"])
            .bind("tui.editor.historyNext", ["ctrl+n"]),
    ));

    let tui = tui_support::new_editor_test_tui(80, 24);
    let editor = Editor::new(&tui, tui_support::default_editor_theme());

    editor.add_to_history("older prompt");
    editor.add_to_history("newer\nmultiline prompt");
    editor.set_text("draft");
    editor.handle_input("\x1b[D");
    editor.handle_input("\x1b[D");

    editor.handle_input("\x10"); // Ctrl+P
    assert_eq!(editor.get_text(), "newer\nmultiline prompt");
    assert_eq!(editor.get_cursor().line, 0);
    assert_eq!(editor.get_cursor().col, 0);

    editor.handle_input("\x10"); // Ctrl+P
    assert_eq!(editor.get_text(), "older prompt");

    editor.handle_input("\x0e"); // Ctrl+N
    assert_eq!(editor.get_text(), "newer\nmultiline prompt");
    assert_eq!((editor.get_cursor().line, editor.get_cursor().col), (1, 16));

    editor.handle_input("\x0e"); // Ctrl+N
    assert_eq!(editor.get_text(), "draft");
    assert_eq!((editor.get_cursor().line, editor.get_cursor().col), (0, 3));
}
