//! Port of `test/scrollbar-theme.test.ts` at pin `60e7e76`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::path::{Path, PathBuf};

use pi_coding_agent::modes::interactive::theme::theme::{ColorMode, load_theme_from_path};
use serde_json::{Value, json};
use tempfile::TempDir;

const DARK: &str = include_str!("../src/modes/interactive/theme/dark.json");

fn dark_theme() -> Value {
    serde_json::from_str(DARK).expect("dark theme")
}

fn write_theme(root: &Path, name: &str, theme: &Value) -> PathBuf {
    let path = root.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string(theme).expect("json")).expect("write");
    path
}

#[test]
fn scrollbar_tokens_fall_back_to_muted_and_text() {
    for (token, fallback) in [("scrollbarTrack", "muted"), ("scrollbarThumb", "text")] {
        let root = TempDir::new().expect("temp");
        let mut theme_json = dark_theme();
        theme_json["name"] = json!(format!("missing-{token}-theme"));
        theme_json["colors"]
            .as_object_mut()
            .expect("colors")
            .remove(token);
        let path = write_theme(root.path(), "theme", &theme_json);
        let theme = load_theme_from_path(&path, Some(ColorMode::Truecolor)).expect("theme");
        assert_eq!(theme.get_fg_ansi(token), theme.get_fg_ansi(fallback));
    }
}

#[test]
fn explicit_scrollbar_colors_win() {
    let root = TempDir::new().expect("temp");
    let mut theme_json = dark_theme();
    theme_json["name"] = json!("custom-scrollbar-theme");
    theme_json["colors"]["scrollbarTrack"] = json!("#654321");
    theme_json["colors"]["scrollbarThumb"] = json!("#123456");
    let path = write_theme(root.path(), "theme", &theme_json);
    let theme = load_theme_from_path(&path, Some(ColorMode::Truecolor)).expect("theme");
    assert_eq!(theme.get_fg_ansi("scrollbarTrack"), "\x1b[38;2;101;67;33m");
    assert_eq!(theme.get_fg_ansi("scrollbarThumb"), "\x1b[38;2;18;52;86m");
}

#[test]
fn search_highlights_fall_back_to_selection_and_text() {
    let root = TempDir::new().expect("temp");
    let mut theme_json = dark_theme();
    theme_json["name"] = json!("legacy-search-theme");
    theme_json["colors"]
        .as_object_mut()
        .expect("colors")
        .remove("searchMatchBg");
    theme_json["colors"]
        .as_object_mut()
        .expect("colors")
        .remove("searchMatchText");
    let path = write_theme(root.path(), "theme", &theme_json);
    let theme = load_theme_from_path(&path, Some(ColorMode::Truecolor)).expect("theme");
    assert_eq!(
        theme.get_bg_ansi("searchMatchBg"),
        theme.get_bg_ansi("selectedBg")
    );
    assert_eq!(
        theme.get_fg_ansi("searchMatchText"),
        theme.get_fg_ansi("text")
    );
}

#[test]
fn explicit_search_highlights_win() {
    let root = TempDir::new().expect("temp");
    let mut theme_json = dark_theme();
    theme_json["name"] = json!("custom-search-theme");
    theme_json["colors"]["searchMatchBg"] = json!("#112233");
    theme_json["colors"]["searchMatchText"] = json!("#223344");
    let path = write_theme(root.path(), "theme", &theme_json);
    let theme = load_theme_from_path(&path, Some(ColorMode::Truecolor)).expect("theme");
    assert_eq!(theme.get_bg_ansi("searchMatchBg"), "\x1b[48;2;17;34;51m");
    assert_eq!(theme.get_fg_ansi("searchMatchText"), "\x1b[38;2;34;51;68m");
}
