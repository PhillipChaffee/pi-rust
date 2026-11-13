//! The extension editor suite, upstream's `extension-editor.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759` (no upstream test file), plus
//! the `external-editor.ts` cases from `test/external-editor.test.ts` that
//! the inline port carries.
//!
//! The external-editor tests drive `/bin/sh` fixtures; the success, BOM,
//! empty, failure, and cleanup paths match upstream's fake-editor fixture.

#![expect(
    clippy::expect_used,
    reason = "tests pin outcomes; an unexpected result panics by design"
)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pi_coding_agent::core::keybindings::AppKeybindingsManager;
use pi_coding_agent::modes::interactive::components::extension_editor::ExtensionEditorComponent;
use pi_coding_agent::modes::interactive::external_editor::{
    ExternalEditorResult, edit_in_external_editor_blocking, resolve_external_editor_command_with,
};
use pi_coding_agent::modes::interactive::theme::theme::init_theme;
use pi_coding_agent::utils::ansi::strip_ansi;
use pi_tui::components::editor::EditorOptions;
use pi_tui::tui::{Component, Focusable, Tui, TuiConfig};
use pi_tui::utils::visible_width;
use tempfile::TempDir;

type Callbacks = (Rc<RefCell<Vec<String>>>, Rc<RefCell<usize>>);

fn new_component(
    title: &str,
    prefill: Option<&str>,
    command: Option<&str>,
) -> (Rc<ExtensionEditorComponent>, Callbacks) {
    init_theme(None, false);
    let dir = TempDir::new().expect("temp dir");
    let manager = Rc::new(AppKeybindingsManager::create(dir.path()));
    let tui: Rc<Tui> = Tui::new(TuiConfig::default());
    let submitted: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let cancelled = Rc::new(RefCell::new(0));

    let on_submit = {
        let submitted = Rc::clone(&submitted);
        Rc::new(move |value: &str| submitted.borrow_mut().push(value.to_string()))
    };
    let on_cancel = {
        let cancelled = Rc::clone(&cancelled);
        Rc::new(move || *cancelled.borrow_mut() += 1)
    };
    let component = ExtensionEditorComponent::new(
        tui,
        manager,
        title,
        prefill,
        on_submit,
        on_cancel,
        EditorOptions::default(),
        command,
    );
    (Rc::new(component), (submitted, cancelled))
}

fn rendered(component: &ExtensionEditorComponent, width: usize) -> String {
    component
        .render(width)
        .iter()
        .map(|line| strip_ansi(line))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn renders_borders_title_editor_and_hint() {
    let (component, _) = new_component("Extension prompt", None, None);
    let lines = component.render(40);
    assert_eq!(strip_ansi(&lines[0]), "─".repeat(40));
    assert_eq!(
        strip_ansi(lines.last().expect("bottom border")),
        "─".repeat(40)
    );

    let text = rendered(&component, 40);
    assert!(text.contains("Extension prompt"));
    assert!(text.contains("submit"));
    assert!(text.contains("cancel"));
    assert!(text.contains("external editor"));
    for (index, line) in lines.iter().enumerate() {
        assert!(
            visible_width(line) <= 40,
            "line {index} exceeds width: {:?}",
            strip_ansi(line)
        );
    }
}

#[test]
fn prefill_populates_the_editor() {
    let (component, _) = new_component("Extension prompt", Some("seed text"), None);
    assert_eq!(component.editor().get_text(), "seed text");
}

#[test]
fn enter_submits_the_editor_text() {
    let (component, (submitted, _)) = new_component("Extension prompt", None, None);
    for character in "hi".chars() {
        component.handle_input(&character.to_string());
    }
    component.handle_input("\r");

    assert_eq!(submitted.borrow().as_slice(), ["hi"]);
}

#[test]
fn escape_cancels() {
    let (component, (submitted, cancelled)) = new_component("Extension prompt", None, None);
    component.handle_input("\x1b");

    assert_eq!(*cancelled.borrow(), 1);
    assert!(submitted.borrow().is_empty());
}

#[test]
fn long_titles_keep_every_line_within_width() {
    let (component, _) = new_component(&"한글".repeat(20), None, None);
    // Wide glyphs are two cells, so one-cell widths cannot hold the title;
    // the invariant starts at the smallest width a grapheme fits.
    for width in [2, 10, 20, 40] {
        for line in component.render(width) {
            assert!(
                visible_width(&line) <= width,
                "width {width}: {:?}",
                strip_ansi(&line)
            );
        }
    }
}

#[test]
fn focus_reaches_the_wrapped_editor() {
    let (component, _) = new_component("Extension prompt", None, None);
    component.set_focused(true);
    assert!(component.is_focused());
    assert!(component.editor().is_focused());
    component.set_focused(false);
    assert!(!component.editor().is_focused());
}

#[test]
fn command_resolution_prefers_explicit_then_visual_then_editor_then_default() {
    let env = |key: &str| match key {
        "VISUAL" => Some("vi".to_string()),
        "EDITOR" => Some("ed".to_string()),
        _ => None,
    };
    assert_eq!(
        resolve_external_editor_command_with(Some("code -w"), &env, false),
        "code -w"
    );
    assert_eq!(
        resolve_external_editor_command_with(Some(""), &env, false),
        "vi"
    );

    let editor_only = |key: &str| (key == "EDITOR").then(|| "ed".to_string());
    assert_eq!(
        resolve_external_editor_command_with(None, &editor_only, false),
        "ed"
    );

    let empty = |_: &str| None;
    assert_eq!(
        resolve_external_editor_command_with(None, &empty, false),
        "nano"
    );
    assert_eq!(
        resolve_external_editor_command_with(None, &empty, true),
        "notepad"
    );
}

#[cfg(unix)]
fn write_script(dir: &Path, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;

    let path = dir.join(name);
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write script");
    let mut permissions = std::fs::metadata(&path)
        .expect("script metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&path, permissions).expect("chmod script");
    path
}

#[cfg(unix)]
#[test]
fn external_editor_reads_the_edited_file() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_script(dir.path(), "edit.sh", "printf 'edited' > \"$1\"");
    let result = edit_in_external_editor_blocking(&script.display().to_string(), "original");
    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: "edited".to_string()
        }
    );
}

#[cfg(unix)]
#[test]
fn external_editor_strips_the_bom_and_one_final_newline() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_script(
        dir.path(),
        "edit.sh",
        "printf '\\357\\273\\277edited\\n' > \"$1\"",
    );
    let result = edit_in_external_editor_blocking(&script.display().to_string(), "original");
    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: "edited".to_string()
        }
    );
}

#[cfg(unix)]
#[test]
fn external_editor_returns_empty_content_when_cleared() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_script(dir.path(), "edit.sh", ": > \"$1\"");
    let result = edit_in_external_editor_blocking(&script.display().to_string(), "original");
    assert_eq!(
        result,
        ExternalEditorResult::Complete {
            content: String::new()
        }
    );
}

#[cfg(unix)]
#[test]
fn external_editor_failure_reports_failed() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_script(dir.path(), "edit.sh", "exit 3");
    assert_eq!(
        edit_in_external_editor_blocking(&script.display().to_string(), "original"),
        ExternalEditorResult::Failed
    );
    assert_eq!(
        edit_in_external_editor_blocking("/definitely/not/an/editor", "original"),
        ExternalEditorResult::Failed
    );
}

#[cfg(unix)]
#[test]
fn external_editor_removes_its_private_directory() {
    let dir = TempDir::new().expect("temp dir");
    let capture = dir.path().join("capture.txt");
    let script = write_script(
        dir.path(),
        "edit.sh",
        "printf '%s' \"$(dirname \"$2\")\" > \"$1\"\nprintf 'edited' > \"$2\"",
    );
    let command = format!("{} {}", script.display(), capture.display());
    let result = edit_in_external_editor_blocking(&command, "original");
    assert!(matches!(result, ExternalEditorResult::Complete { .. }));

    let saved_directory = std::fs::read_to_string(&capture).expect("captured directory");
    assert!(
        !Path::new(&saved_directory).exists(),
        "temporary directory survived: {saved_directory}"
    );
}

#[cfg(unix)]
#[test]
fn open_external_editor_reloads_the_prompt() {
    let dir = TempDir::new().expect("temp dir");
    let script = write_script(dir.path(), "edit.sh", "printf 'from editor' > \"$1\"");
    let (component, _) = new_component(
        "Extension prompt",
        Some("draft"),
        Some(&script.display().to_string()),
    );

    component.open_external_editor();
    assert_eq!(component.editor().get_text(), "from editor");
}
