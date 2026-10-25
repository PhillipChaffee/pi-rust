//! Unit tests for the theme color engine (#187).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use indexmap::IndexMap;
use pi_tui::terminal_colors::RgbColor;

use super::super::theme_json::ValidatedThemeJson;
use super::{
    ColorMode, Theme, ThemeColorValue, ThemeOptions, ansi256_to_hex, bg_ansi, bold, create_theme,
    fg_ansi, get_language_from_path, hex_to_256, hex_to_rgb, inverse, italic,
    parse_auto_theme_setting, resolve_theme_colors, resolve_theme_setting, resolve_var_refs,
    rgb_to_256, strikethrough, underline,
};

fn color(text: &str) -> ThemeColorValue {
    ThemeColorValue::Text(text.to_string())
}

fn map(entries: &[(&str, ThemeColorValue)]) -> IndexMap<String, ThemeColorValue> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn theme_with(fg: &[(&str, ThemeColorValue)], bg: &[(&str, ThemeColorValue)]) -> Theme {
    Theme::new(
        map(fg),
        map(bg),
        ColorMode::Truecolor,
        ThemeOptions::default(),
    )
}

#[test]
fn parses_hex_colors() {
    assert_eq!(
        hex_to_rgb("#12ab34"),
        Ok(RgbColor {
            r: 0x12,
            g: 0xab,
            b: 0x34
        })
    );
    assert_eq!(
        hex_to_rgb("12ab34").expect("no hash"),
        RgbColor {
            r: 0x12,
            g: 0xab,
            b: 0x34
        }
    );
    assert_eq!(
        hex_to_rgb("#12345"),
        Err("Invalid hex color: #12345".to_string())
    );
    assert_eq!(
        hex_to_rgb("#gggggg"),
        Err("Invalid hex color: #gggggg".to_string())
    );
}

#[test]
fn encodes_256_color_values() {
    assert_eq!(
        fg_ansi(&ThemeColorValue::Index(24), ColorMode::Truecolor),
        "\x1b[38;5;24m"
    );
    assert_eq!(
        bg_ansi(&ThemeColorValue::Index(24), ColorMode::Truecolor),
        "\x1b[48;5;24m"
    );
    assert_eq!(
        fg_ansi(&color("#123456"), ColorMode::Truecolor),
        "\x1b[38;2;18;52;86m"
    );
    assert_eq!(
        bg_ansi(&color("#123456"), ColorMode::Truecolor),
        "\x1b[48;2;18;52;86m"
    );
    assert_eq!(fg_ansi(&color(""), ColorMode::Truecolor), "\x1b[39m");
    assert_eq!(bg_ansi(&color(""), ColorMode::Truecolor), "\x1b[49m");
}

#[test]
fn snaps_hex_colors_to_the_256_palette() {
    assert_eq!(ansi256_to_hex(0), "#000000");
    assert_eq!(ansi256_to_hex(15), "#ffffff");
    assert_eq!(ansi256_to_hex(24), "#005f87");
    assert_eq!(ansi256_to_hex(231), "#ffffff");
    assert_eq!(ansi256_to_hex(232), "#080808");
    assert_eq!(ansi256_to_hex(255), "#eeeeee");
    assert_eq!(hex_to_256("#005f87"), Ok(24));
    // Nearly neutral colors prefer the gray ramp, tinted ones the cube.
    assert_eq!(rgb_to_256(RgbColor { r: 8, g: 8, b: 8 }), 232);
    assert_eq!(
        rgb_to_256(RgbColor {
            r: 200,
            g: 10,
            b: 10
        }),
        160
    );
}

#[test]
fn styles_text_with_chalk_resets() {
    assert_eq!(bold("x"), "\x1b[1mx\x1b[22m");
    assert_eq!(italic("x"), "\x1b[3mx\x1b[23m");
    assert_eq!(underline("x"), "\x1b[4mx\x1b[24m");
    assert_eq!(inverse("x"), "\x1b[7mx\x1b[27m");
    assert_eq!(strikethrough("x"), "\x1b[9mx\x1b[29m");
}

#[test]
fn fills_optional_tokens_and_decorates_text() {
    let theme = theme_with(
        &[
            ("muted", color("#111111")),
            ("text", color("#222222")),
            ("thinkingXhigh", color("#333333")),
            ("accent", color("")),
        ],
        &[("selectedBg", color("#444444"))],
    );
    assert_eq!(
        theme.get_fg_ansi("scrollbarTrack"),
        theme.get_fg_ansi("muted")
    );
    assert_eq!(
        theme.get_fg_ansi("scrollbarThumb"),
        theme.get_fg_ansi("text")
    );
    assert_eq!(
        theme.get_fg_ansi("thinkingMax"),
        theme.get_fg_ansi("thinkingXhigh")
    );
    assert_eq!(
        theme.get_fg_ansi("searchMatchText"),
        theme.get_fg_ansi("text")
    );
    assert_eq!(
        theme.get_bg_ansi("searchMatchBg"),
        theme.get_bg_ansi("selectedBg")
    );
    assert_eq!(theme.fg("accent", "x"), "\x1b[39mx\x1b[39m");
    assert_eq!(theme.bg("selectedBg", "x"), "\x1b[48;2;68;68;68mx\x1b[49m");
    assert_eq!(theme.color_mode(), ColorMode::Truecolor);
    assert_eq!(theme.name(), None);
    assert_eq!(theme.source_path(), None);
    assert!(format!("{theme:?}").starts_with("Theme"));
}

#[test]
fn drops_an_absent_theme_watcher() {
    super::drop_theme_watcher();
}

#[test]
#[should_panic(expected = "Unknown theme color: missing")]
fn unknown_foreground_panics() {
    let theme = theme_with(&[], &[]);
    let _ = theme.fg("missing", "x");
}

#[test]
#[should_panic(expected = "Unknown theme background color: missing")]
fn unknown_background_panics() {
    let theme = theme_with(&[], &[]);
    let _ = theme.bg("missing", "x");
}

#[test]
#[should_panic(expected = "Invalid color value: not-a-color")]
fn unresolved_color_panics_at_construction() {
    let _ = theme_with(&[("accent", color("not-a-color"))], &[]);
}

#[test]
#[should_panic(expected = "Invalid hex color: #zzzzzz")]
fn malformed_hex_panics_at_construction() {
    let _ = theme_with(&[("accent", color("#zzzzzz"))], &[]);
}

#[test]
fn resolves_variable_references() {
    let vars = map(&[
        ("primary", color("secondary")),
        ("secondary", color("#abcdef")),
        ("index", ThemeColorValue::Index(24)),
    ]);
    let mut visited = std::collections::BTreeSet::new();
    assert_eq!(
        resolve_var_refs(&color("primary"), &vars, &mut visited),
        Ok(color("#abcdef"))
    );
    let mut visited = std::collections::BTreeSet::new();
    assert_eq!(
        resolve_var_refs(&ThemeColorValue::Index(24), &vars, &mut visited),
        Ok(ThemeColorValue::Index(24))
    );
    let mut visited = std::collections::BTreeSet::new();
    assert_eq!(
        resolve_var_refs(&color(""), &vars, &mut visited),
        Ok(color(""))
    );
    let mut visited = std::collections::BTreeSet::new();
    assert_eq!(
        resolve_var_refs(&color("absent"), &vars, &mut visited),
        Err("Variable reference not found: absent".to_string())
    );
}

#[test]
fn rejects_circular_variable_references() {
    let vars = map(&[("a", color("b")), ("b", color("a"))]);
    assert_eq!(
        resolve_var_refs(&color("a"), &vars, &mut std::collections::BTreeSet::new()),
        Err("Circular variable reference detected: a".to_string())
    );
}

#[test]
fn resolves_every_color_in_a_map() {
    let vars = map(&[("accent", color("#abcdef"))]);
    let colors = map(&[("accent", color("accent")), ("text", color(""))]);
    let resolved = resolve_theme_colors(&colors, &vars).expect("resolves");
    assert_eq!(resolved.get("accent"), Some(&color("#abcdef")));
    assert_eq!(resolved.get("text"), Some(&color("")));
}

#[test]
fn builds_a_theme_with_fallbacks_and_variables() {
    let json = ValidatedThemeJson {
        name: "custom".to_string(),
        vars: map(&[("primary", color("#123456"))]),
        colors: map(&[("accent", color("primary")), ("text", color(""))]),
        export: None,
    };
    let theme = create_theme(
        &json,
        Some(ColorMode::Color256),
        Some("custom.json".to_string()),
    )
    .expect("theme");
    assert_eq!(theme.name(), Some("custom"));
    assert_eq!(theme.source_path(), Some("custom.json"));
    assert_eq!(theme.color_mode(), ColorMode::Color256);
    assert_eq!(theme.get_fg_ansi("accent"), "\x1b[38;5;23m");
    assert_eq!(theme.get_fg_ansi("text"), "\x1b[39m");
}

#[test]
fn picks_the_color_mode_from_terminal_capabilities() {
    use pi_tui::terminal_image::{
        TerminalCapabilities, reset_capabilities_cache, set_capabilities,
    };
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: false,
        hyperlinks: false,
    });
    let json = ValidatedThemeJson {
        name: "dark".to_string(),
        vars: IndexMap::new(),
        colors: map(&[("accent", color("#123456"))]),
        export: None,
    };
    let theme = create_theme(&json, None, None).expect("theme");
    assert_eq!(theme.color_mode(), ColorMode::Color256);
    set_capabilities(TerminalCapabilities {
        images: None,
        true_color: true,
        hyperlinks: false,
    });
    let theme = create_theme(&json, None, None).expect("theme");
    assert_eq!(theme.color_mode(), ColorMode::Truecolor);
    reset_capabilities_cache();
}

#[test]
fn parses_automatic_theme_settings() {
    assert_eq!(
        parse_auto_theme_setting(Some("light/dark")),
        Some(super::AutoTheme {
            light_theme: "light".to_string(),
            dark_theme: "dark".to_string(),
        })
    );
    assert_eq!(
        parse_auto_theme_setting(Some(" light / dark ")),
        Some(super::AutoTheme {
            light_theme: "light".to_string(),
            dark_theme: "dark".to_string(),
        })
    );
    for setting in [
        None,
        Some(""),
        Some("dark"),
        Some("light/dark/extra"),
        Some("/dark"),
        Some("light/"),
    ] {
        assert_eq!(parse_auto_theme_setting(setting), None, "{setting:?}");
    }
}

#[test]
fn resolves_theme_settings_per_terminal_theme() {
    assert_eq!(
        resolve_theme_setting(Some("light/dark"), super::TerminalTheme::Dark),
        Some("dark".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("light/dark"), super::TerminalTheme::Light),
        Some("light".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("custom"), super::TerminalTheme::Light),
        Some("custom".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("a/b/c"), super::TerminalTheme::Light),
        None
    );
    assert_eq!(
        resolve_theme_setting(None, super::TerminalTheme::Light),
        None
    );
}

#[test]
fn maps_paths_to_languages() {
    assert_eq!(get_language_from_path("main.rs"), Some("rust"));
    assert_eq!(get_language_from_path("app.TSX"), Some("typescript"));
    assert_eq!(get_language_from_path("Dockerfile"), Some("dockerfile"));
    assert_eq!(get_language_from_path("notes.unknown"), None);
    assert_eq!(get_language_from_path("no-extension"), None);
}
