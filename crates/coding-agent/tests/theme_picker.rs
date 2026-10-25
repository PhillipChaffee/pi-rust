//! Port of `test/theme-picker.test.ts` at pin `60e7e76`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::path::PathBuf;

use indexmap::IndexMap;
use pi_coding_agent::modes::interactive::theme::theme::{
    ColorMode, Theme, ThemeOptions, get_available_themes, get_available_themes_with_paths,
    get_available_themes_with_paths_with, set_registered_themes, set_theme_dirs_override,
};
use pi_coding_agent::modes::interactive::theme::theme_json::ThemeColorValue;
use serde_json::{Value, json};
use tempfile::TempDir;

const DARK: &str = include_str!("../src/modes/interactive/theme/dark.json");

fn dark_theme() -> Value {
    serde_json::from_str(DARK).expect("dark theme")
}

fn temp_dirs() -> (TempDir, PathBuf, PathBuf) {
    let root = TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin.clone(), custom.clone())));
    (root, builtin, custom)
}

#[test]
fn uses_custom_theme_content_names_instead_of_file_names() {
    let (_root, _builtin, custom) = temp_dirs();
    let mut theme = dark_theme();
    theme["name"] = json!("bar");
    let path = custom.join("foo.json");
    std::fs::write(&path, serde_json::to_string(&theme).expect("json")).expect("write");

    assert!(get_available_themes().contains(&"bar".to_string()));
    assert!(!get_available_themes().contains(&"foo".to_string()));
    let info = get_available_themes_with_paths();
    assert!(info.contains(
        &pi_coding_agent::modes::interactive::theme::theme::ThemeInfo {
            name: "bar".to_string(),
            path: Some(path.to_string_lossy().into_owned()),
        }
    ));
    set_theme_dirs_override(None);
}

#[test]
fn lists_the_builtins_with_their_reported_paths() {
    let (_root, builtin, custom) = temp_dirs();
    let info = get_available_themes_with_paths_with(&builtin, &custom);
    let names: Vec<&str> = info.iter().map(|entry| entry.name.as_str()).collect();
    assert_eq!(names, vec!["dark", "light"]);
    assert_eq!(
        info[0].path,
        Some(builtin.join("dark.json").to_string_lossy().into_owned())
    );
}

#[test]
fn includes_registered_themes_and_skips_invalid_files() {
    let (_root, builtin, custom) = temp_dirs();
    std::fs::write(custom.join("broken.json"), "{").expect("write");
    let theme = Theme::new(
        IndexMap::<String, ThemeColorValue>::new(),
        IndexMap::<String, ThemeColorValue>::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("registered".to_string()),
            source_path: Some("/tmp/registered.json".to_string()),
            source_info: None,
        },
    );
    set_registered_themes(vec![std::sync::Arc::new(theme)]).expect("register");

    let info = get_available_themes_with_paths_with(&builtin, &custom);
    let registered = info
        .iter()
        .find(|entry| entry.name == "registered")
        .expect("registered theme listed");
    assert_eq!(registered.path.as_deref(), Some("/tmp/registered.json"));
    assert!(!info.iter().any(|entry| entry.name == "broken"));
    set_registered_themes(Vec::new()).expect("clear");
    set_theme_dirs_override(None);
}

#[test]
fn rejects_registered_names_with_slashes() {
    let theme = Theme::new(
        IndexMap::new(),
        IndexMap::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("light/dark".to_string()),
            source_path: None,
            source_info: None,
        },
    );
    let error = set_registered_themes(vec![std::sync::Arc::new(theme)]).expect_err("slash");
    assert!(error.contains("theme names cannot contain \"/\""));
    set_registered_themes(Vec::new()).expect("clear");
}
