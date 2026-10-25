//! Theme loading, switching, and the custom-theme watcher (#187).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use pi_coding_agent::modes::interactive::theme::theme::{
    ColorMode, Theme, ThemeOptions, current_theme_name, get_available_themes, get_theme_by_name,
    init_theme, load_theme, load_theme_from_path, load_theme_json, on_theme_change,
    set_registered_themes, set_theme, set_theme_dirs_override, set_theme_instance,
    stop_theme_watcher, theme,
};
use pi_coding_agent::modes::interactive::theme::theme_json::ThemeColorValue;
use serde_json::{Value, json};
use tempfile::TempDir;

const DARK: &str = include_str!("../src/modes/interactive/theme/dark.json");

fn dark_theme() -> Value {
    serde_json::from_str(DARK).expect("dark theme")
}

fn temp_dirs() -> (TempDir, PathBuf) {
    let root = TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin, custom.clone())));
    (root, custom)
}

fn write_theme(custom: &std::path::Path, name: &str, accent: &str) -> PathBuf {
    let mut value = dark_theme();
    value["name"] = json!(name);
    value["colors"]["accent"] = json!(accent);
    let path = custom.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string(&value).expect("json")).expect("write");
    path
}

#[test]
#[should_panic(expected = "Theme not initialized. Call init_theme() first.")]
fn theme_accessor_panics_before_init() {
    let _ = theme();
}

#[test]
fn init_loads_a_builtin_and_falls_back_to_dark() {
    let (_root, _custom) = temp_dirs();
    init_theme(Some("light"), false);
    assert_eq!(theme().name(), Some("light"));
    assert_eq!(current_theme_name().as_deref(), Some("light"));

    init_theme(Some("missing-theme"), false);
    assert_eq!(theme().name(), Some("dark"));
    assert_eq!(current_theme_name().as_deref(), Some("dark"));
    set_theme_dirs_override(None);
}

#[test]
fn init_without_a_name_uses_detected_appearance() {
    let (_root, _custom) = temp_dirs();
    init_theme(None, false);
    let name = theme().name().expect("name").to_string();
    assert!(name == "dark" || name == "light");
    set_theme_dirs_override(None);
}

#[test]
fn set_theme_reports_failures_and_notifies() {
    let (_root, _custom) = temp_dirs();
    let calls = Arc::new(Mutex::new(0_u32));
    let counter = Arc::clone(&calls);
    on_theme_change(Arc::new(move || {
        *counter.lock().expect("counter") += 1;
    }));

    let result = set_theme("light", false);
    assert!(result.success);
    assert_eq!(result.error, None);
    assert_eq!(theme().name(), Some("light"));

    let result = set_theme("missing-theme", false);
    assert!(!result.success);
    assert!(
        result
            .error
            .expect("error")
            .contains("Theme not found: missing-theme")
    );
    assert_eq!(theme().name(), Some("dark"));
    assert_eq!(current_theme_name().as_deref(), Some("dark"));
    assert_eq!(*calls.lock().expect("calls"), 1);
    set_theme_dirs_override(None);
}

#[test]
fn custom_themes_load_by_name_from_the_agent_directory() {
    let (_root, custom) = temp_dirs();
    write_theme(&custom, "mytheme", "#112233");
    init_theme(Some("mytheme"), false);
    assert_eq!(theme().get_fg_ansi("accent"), "\x1b[38;2;17;34;51m");
    assert_eq!(
        get_theme_by_name("mytheme").expect("theme").name(),
        Some("mytheme")
    );
    assert!(get_theme_by_name("absent").is_none());
    set_theme_dirs_override(None);
}

#[test]
fn registered_themes_win_over_disk_lookups() {
    let (_root, custom) = temp_dirs();
    let registered_path = write_theme(&custom, "registered-json", "#445566");
    let theme_instance = Theme::new(
        IndexMap::from([(
            "accent".to_string(),
            ThemeColorValue::Text("#abcdef".to_string()),
        )]),
        IndexMap::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("registered".to_string()),
            source_path: Some(registered_path.to_string_lossy().into_owned()),
            source_info: None,
        },
    );
    set_registered_themes(vec![Arc::new(theme_instance)]).expect("register");

    let loaded = load_theme("registered", None).expect("registered");
    assert!(loaded.get_fg_ansi("accent").starts_with("\x1b[38;"));
    let json = load_theme_json("registered").expect("source json");
    assert_eq!(json.name, "registered-json");

    let sourceless = Theme::new(
        IndexMap::new(),
        IndexMap::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("sourceless".to_string()),
            source_path: None,
            source_info: None,
        },
    );
    set_registered_themes(vec![Arc::new(sourceless)]).expect("register");
    let error = load_theme_json("sourceless").expect_err("no source");
    assert!(error.contains("does not have a source path for export"));
    set_registered_themes(Vec::new()).expect("clear");
    set_theme_dirs_override(None);
}

#[test]
fn set_theme_instance_replaces_the_active_theme() {
    let (_root, _custom) = temp_dirs();
    init_theme(Some("dark"), false);
    let instance = Theme::new(
        IndexMap::from([(
            "accent".to_string(),
            ThemeColorValue::Text("#010203".to_string()),
        )]),
        IndexMap::new(),
        ColorMode::Truecolor,
        ThemeOptions {
            name: Some("in-memory".to_string()),
            source_path: None,
            source_info: None,
        },
    );
    set_theme_instance(Arc::new(instance));
    assert_eq!(current_theme_name().as_deref(), Some("<in-memory>"));
    assert_eq!(theme().name(), Some("in-memory"));
    assert!(theme().get_fg_ansi("accent").starts_with("\x1b[38;"));
    set_theme_dirs_override(None);
}

#[test]
fn load_from_path_reports_read_parse_and_validation_errors() {
    let root = TempDir::new().expect("temp");
    let missing = root.path().join("missing.json");
    let error = load_theme_from_path(&missing, None).expect_err("missing");
    assert!(error.starts_with("Failed to read theme "));

    let invalid_json = root.path().join("invalid.json");
    std::fs::write(&invalid_json, "{").expect("write");
    let error = load_theme_from_path(&invalid_json, None).expect_err("json");
    assert!(error.starts_with("Failed to parse theme "));

    let mut invalid_theme = dark_theme();
    invalid_theme["colors"]
        .as_object_mut()
        .expect("colors")
        .remove("accent");
    let invalid_theme_path = root.path().join("invalid-theme.json");
    std::fs::write(
        &invalid_theme_path,
        serde_json::to_string(&invalid_theme).expect("json"),
    )
    .expect("write");
    let error = load_theme_from_path(&invalid_theme_path, None).expect_err("validation");
    assert!(error.contains("Missing required color tokens"));

    let bom_path = root.path().join("bom.json");
    std::fs::write(&bom_path, format!("\u{feff}{DARK}")).expect("write");
    let loaded = load_theme_from_path(&bom_path, Some(ColorMode::Truecolor)).expect("bom");
    assert_eq!(loaded.name(), Some("dark"));
}

#[test]
fn available_themes_sort_and_dedupe_by_content_name() {
    let (_root, custom) = temp_dirs();
    let mut shadow = dark_theme();
    shadow["name"] = json!("dark");
    std::fs::write(
        custom.join("shadow.json"),
        serde_json::to_string(&shadow).expect("json"),
    )
    .expect("write");
    write_theme(&custom, "aaa-theme", "#000000");
    let names = get_available_themes();
    assert_eq!(names, vec!["aaa-theme", "dark", "light"]);
    set_theme_dirs_override(None);
}

#[test]
fn watcher_reloads_edits_and_stops() {
    let (_root, custom) = temp_dirs();
    let path = write_theme(&custom, "watched", "#112233");
    init_theme(Some("watched"), true);
    assert!(theme().get_fg_ansi("accent").starts_with("\x1b[38;"));

    let mut edited = dark_theme();
    edited["name"] = json!("watched");
    edited["colors"]["accent"] = json!("#445566");
    std::fs::write(&path, serde_json::to_string(&edited).expect("json")).expect("write");
    // The reload builds at the same terminal capabilities, so the expected
    // escape comes from a fresh load rather than a hard-coded truecolor value.
    let expected = load_theme_from_path(&path, None)
        .expect("edited theme")
        .get_fg_ansi("accent")
        .to_string();
    let deadline = Instant::now() + Duration::from_secs(5);
    while theme().get_fg_ansi("accent") != expected && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(theme().get_fg_ansi("accent"), expected);

    stop_theme_watcher();
    edited["colors"]["accent"] = json!("#778899");
    std::fs::write(&path, serde_json::to_string(&edited).expect("json")).expect("write");
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(theme().get_fg_ansi("accent"), expected);
    set_theme_dirs_override(None);
}

#[test]
fn rejects_circular_theme_variables() {
    let root = TempDir::new().expect("temp");
    let mut theme_json = dark_theme();
    theme_json["vars"]["a"] = json!("b");
    theme_json["vars"]["b"] = json!("a");
    theme_json["colors"]["accent"] = json!("a");
    let path = root.path().join("cyclic.json");
    std::fs::write(&path, serde_json::to_string(&theme_json).expect("json")).expect("write");
    let error = load_theme_from_path(&path, None).expect_err("cycle");
    assert_eq!(error, "Circular variable reference detected: a");
}
