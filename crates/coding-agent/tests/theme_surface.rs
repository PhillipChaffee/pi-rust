//! TUI theme getters, highlight wiring, and export color resolution (#187).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_agent_core::types::ThinkingLevel;
use pi_coding_agent::export_html::ThemeColorsSource;
use pi_coding_agent::modes::interactive::theme::theme::{
    TerminalTheme, ThemeSystemSource, get_bash_mode_border_color, get_default_theme,
    get_editor_theme, get_markdown_theme, get_resolved_theme_colors, get_select_list_theme,
    get_settings_list_theme, get_thinking_border_color, highlight_code, init_theme, is_light_theme,
    set_theme_dirs_override,
};

#[test]
fn markdown_theme_decorates_and_highlights() {
    init_theme(Some("dark"), false);
    let theme = get_markdown_theme();
    let heading = (theme.heading)("Title");
    assert!(heading.starts_with("\x1b[38;2;"));
    assert!(heading.ends_with("\x1b[39m"));
    assert_eq!((theme.bold)("x"), "\x1b[1mx\x1b[22m");
    assert_eq!((theme.italic)("x"), "\x1b[3mx\x1b[23m");
    assert_eq!((theme.underline)("x"), "\x1b[4mx\x1b[24m");
    assert_eq!((theme.strikethrough)("x"), "\x1b[9mx\x1b[29m");

    let highlighted = (theme.highlight_code.as_ref().expect("highlighter"))(
        "fn main() {\n    let x = 1;\n}\n",
        Some("rust"),
    );
    assert!(highlighted.len() >= 3);
    assert!(highlighted.join("\n").contains('\u{1b}'));

    let markdown = highlight_code("**strong** and *em*", Some("markdown"));
    assert!(!markdown.is_empty());

    let plain = (theme.highlight_code.as_ref().expect("highlighter"))("text", None);
    assert_eq!(plain.len(), 1);
    assert!(plain[0].starts_with("\x1b[38;2;"));
}

#[test]
fn highlight_code_skips_unknown_languages() {
    init_theme(Some("dark"), false);
    let lines = highlight_code("let x = 1;", Some("not-a-language"));
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with("\x1b[38;2;"));
    assert_eq!(highlight_code("a\nb", None).len(), 2);
}

#[test]
fn tui_theme_getters_decorate() {
    init_theme(Some("dark"), false);
    let select = get_select_list_theme();
    assert!((select.selected_prefix)(">").ends_with("\x1b[39m"));
    assert!((select.description)("d").ends_with("\x1b[39m"));

    let editor = get_editor_theme();
    assert!((editor.border_color)("x").ends_with("\x1b[39m"));

    let settings = get_settings_list_theme();
    assert!((settings.label)("label", true).ends_with("\x1b[39m"));
    assert_eq!((settings.label)("label", false), "label");
    assert!((settings.value)("value", true).ends_with("\x1b[39m"));
    assert!((settings.value)("value", false).ends_with("\x1b[39m"));
    assert!((settings.description)("d").ends_with("\x1b[39m"));
    assert!(settings.cursor.ends_with("\x1b[39m"));
    assert!((settings.hint)("h").ends_with("\x1b[39m"));

    for level in [
        ThinkingLevel::Off,
        ThinkingLevel::Minimal,
        ThinkingLevel::Low,
        ThinkingLevel::Medium,
        ThinkingLevel::High,
        ThinkingLevel::Xhigh,
        ThinkingLevel::Max,
    ] {
        assert!(get_thinking_border_color(level)("x").ends_with("\x1b[39m"));
    }
    assert!(get_bash_mode_border_color()("x").ends_with("\x1b[39m"));
}

#[test]
fn resolves_theme_colors_for_export() {
    init_theme(Some("dark"), false);
    let dark = get_resolved_theme_colors(Some("dark")).expect("dark colors");
    assert_eq!(dark.get("accent").map(String::as_str), Some("#8abeb7"));
    assert_eq!(dark.get("border").map(String::as_str), Some("#5f87ff"));

    let light = get_resolved_theme_colors(Some("light")).expect("light colors");
    assert_eq!(light.get("accent").map(String::as_str), Some("#5a8080"));

    let current = get_resolved_theme_colors(None).expect("current colors");
    assert_eq!(current.get("accent"), dark.get("accent"));
    assert!(get_resolved_theme_colors(Some("missing")).is_err());
    assert!(is_light_theme(Some("light")));
    assert!(!is_light_theme(Some("dark")));
    assert!(!is_light_theme(None));
}

#[test]
fn empty_colors_become_the_default_text_and_indices_become_hex() {
    let root = tempfile::TempDir::new().expect("temp");
    let builtin = root.path().join("builtin");
    let custom = root.path().join("custom");
    std::fs::create_dir_all(&builtin).expect("builtin");
    std::fs::create_dir_all(&custom).expect("custom");
    set_theme_dirs_override(Some((builtin, custom.clone())));
    let mut theme: serde_json::Value =
        serde_json::from_str(include_str!("../src/modes/interactive/theme/light.json"))
            .expect("light");
    theme["name"] = serde_json::json!("mixed");
    theme["colors"]["accent"] = serde_json::json!(24);
    theme["colors"]["text"] = serde_json::json!("");
    std::fs::write(
        custom.join("mixed.json"),
        serde_json::to_string(&theme).expect("json"),
    )
    .expect("write");
    let colors = get_resolved_theme_colors(Some("mixed")).expect("colors");
    assert_eq!(colors.get("accent").map(String::as_str), Some("#005f87"));
    // The empty value takes the fallback; only a theme named `light` reads
    // as light, exactly upstream's name check.
    assert_eq!(colors.get("text").map(String::as_str), Some("#e5e5e7"));
    set_theme_dirs_override(None);
}

#[test]
fn theme_system_source_feeds_the_html_export() {
    init_theme(Some("dark"), false);
    let source = ThemeSystemSource;
    let colors = source.resolved_colors(Some("dark"));
    assert_eq!(colors.get("accent").map(String::as_str), Some("#8abeb7"));
    let export = source.export_colors(Some("dark"));
    assert_eq!(export.page_bg.as_deref(), Some("#18181e"));
    assert_eq!(export.card_bg.as_deref(), Some("#1e1e24"));
    assert_eq!(export.info_bg.as_deref(), Some("#3c3728"));
    let missing = source.resolved_colors(Some("missing"));
    assert!(missing.is_empty());
}

#[test]
fn default_and_named_terminal_themes_agree() {
    let detected = get_default_theme();
    assert!(matches!(detected.theme_name(), "dark" | "light"));
    assert_eq!(TerminalTheme::Dark.theme_name(), "dark");
    assert_eq!(TerminalTheme::Light.theme_name(), "light");
}
