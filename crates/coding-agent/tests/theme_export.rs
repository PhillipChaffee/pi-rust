//! Port of `test/theme-export.test.ts` at pin `60e7e76`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use std::path::{Path, PathBuf};

use pi_coding_agent::modes::interactive::theme::theme::{
    get_theme_export_colors, get_theme_export_colors_with, set_theme_dirs_override,
};
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

fn write_theme(custom: &Path, name: &str, theme: &Value) {
    std::fs::write(
        custom.join(format!("{name}.json")),
        serde_json::to_string(theme).expect("json"),
    )
    .expect("write");
}

#[test]
fn resolves_export_variable_references_like_colors() {
    let (_root, _builtin, custom) = temp_dirs();
    let mut theme = dark_theme();
    theme["name"] = json!("custom-export-vars");
    theme["vars"]["pageBgVar"] = json!("#112233");
    theme["vars"]["pageBgAlias"] = json!("pageBgVar");
    theme["vars"]["infoBgVar"] = json!("#445566");
    theme["vars"]["cardBgVar"] = json!("#223344");
    theme["export"] = json!({
        "pageBg": "pageBgAlias",
        "cardBg": "cardBgVar",
        "infoBg": "infoBgVar"
    });
    write_theme(&custom, "custom-export-vars", &theme);

    let colors = get_theme_export_colors(Some("custom-export-vars"));
    assert_eq!(colors.page_bg.as_deref(), Some("#112233"));
    assert_eq!(colors.card_bg.as_deref(), Some("#223344"));
    assert_eq!(colors.info_bg.as_deref(), Some("#445566"));
    set_theme_dirs_override(None);
}

#[test]
fn resolves_recursive_vars_and_converts_indices_to_hex() {
    let (_root, _builtin, custom) = temp_dirs();
    let mut theme = dark_theme();
    theme["name"] = json!("custom-export-recursive");
    theme["vars"]["deepPageBg"] = json!("#abcdef");
    theme["vars"]["pageBgAlias"] = json!("deepPageBg");
    theme["vars"]["cardBgAnsi"] = json!(24);
    theme["export"] = json!({
        "pageBg": "pageBgAlias",
        "cardBg": "cardBgAnsi",
        "infoBg": ""
    });
    write_theme(&custom, "custom-export-recursive", &theme);

    let colors = get_theme_export_colors(Some("custom-export-recursive"));
    assert_eq!(colors.page_bg.as_deref(), Some("#abcdef"));
    assert_eq!(colors.card_bg.as_deref(), Some("#005f87"));
    assert_eq!(colors.info_bg, None);
    set_theme_dirs_override(None);
}

#[test]
fn missing_themes_and_missing_export_sections_answer_defaults() {
    let (_root, builtin, _custom) = temp_dirs();
    assert_eq!(
        get_theme_export_colors(Some("does-not-exist")),
        pi_coding_agent::export_html::ThemeExportColors::default()
    );
    let mut theme = dark_theme();
    theme.as_object_mut().expect("object").remove("export");
    write_theme(&builtin, "no-export", &theme);
    assert_eq!(
        get_theme_export_colors_with(Some("no-export"), &builtin),
        pi_coding_agent::export_html::ThemeExportColors::default()
    );
    set_theme_dirs_override(None);
}
