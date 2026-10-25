//! Unit tests for theme document validation (#187).

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use serde_json::{Value, json};

use super::{parse_theme_json_content, validate_theme_json};

const DARK: &str = include_str!("../dark.json");
const LIGHT: &str = include_str!("../light.json");

fn dark_json() -> Value {
    serde_json::from_str(DARK).expect("dark theme parses")
}

#[test]
fn accepts_the_builtin_themes() {
    let dark = validate_theme_json("dark", &dark_json()).expect("dark validates");
    assert_eq!(dark.name, "dark");
    assert!(dark.colors.contains_key("accent"));
    assert!(dark.export.is_some());
    let light = validate_theme_json("light", &serde_json::from_str(LIGHT).expect("light json"))
        .expect("light validates");
    assert_eq!(light.name, "light");
}

#[test]
fn reports_one_missing_required_token() {
    let mut theme = dark_json();
    theme["colors"]
        .as_object_mut()
        .expect("colors object")
        .remove("accent");
    let error = validate_theme_json("broken", &theme).expect_err("missing accent");
    assert!(error.starts_with("Invalid theme \"broken\":"));
    assert!(error.contains("Missing required color tokens:\n  - accent"));
    assert!(error.contains("See the built-in themes (dark.json, light.json)"));
}

#[test]
fn reports_every_token_when_colors_is_absent() {
    let theme = json!({ "name": "empty" });
    let error = validate_theme_json("empty", &theme).expect_err("missing colors");
    for token in ["accent", "bashMode", "thinkingXhigh", "toolOutput"] {
        assert!(error.contains(&format!("  - {token}")), "{token} missing");
    }
}

#[test]
fn rejects_a_non_object_document() {
    let error = validate_theme_json("array", &json!([])).expect_err("array");
    assert!(error.contains("Expected an object with a \"colors\" map."));
}

#[test]
fn rejects_a_missing_or_mistyped_name() {
    let mut theme = dark_json();
    theme.as_object_mut().expect("object").remove("name");
    assert!(
        validate_theme_json("no-name", &theme)
            .expect_err("no name")
            .contains("/name: Required")
    );

    let mut theme = dark_json();
    theme["name"] = json!(7);
    assert!(
        validate_theme_json("number-name", &theme)
            .expect_err("bad name")
            .contains("/name: Expected string")
    );
}

#[test]
fn reserves_slash_in_theme_names() {
    let mut theme = dark_json();
    theme["name"] = json!("light/dark");
    let error = validate_theme_json("slash", &theme).expect_err("slash name");
    assert!(error.contains("theme names cannot contain \"/\""));
}

#[test]
fn rejects_mistyped_vars_and_colors_sections() {
    let mut theme = dark_json();
    theme["vars"] = json!([]);
    assert!(
        validate_theme_json("vars", &theme)
            .expect_err("vars array")
            .contains("/vars: Expected object")
    );

    let mut theme = dark_json();
    theme["colors"] = json!("nope");
    assert!(
        validate_theme_json("colors", &theme)
            .expect_err("colors string")
            .contains("/colors: Expected object")
    );
}

#[test]
fn rejects_out_of_range_and_fractional_color_indices() {
    for value in [json!(300), json!(-1), json!(1.5)] {
        let mut theme = dark_json();
        theme["colors"]["accent"] = value;
        let error = validate_theme_json("index", &theme).expect_err("bad index");
        assert!(
            error.contains("/colors/accent: Expected a string or an integer 0-255"),
            "{error}"
        );
    }
}

#[test]
fn accepts_palette_bounds_and_unknown_tokens() {
    let mut theme = dark_json();
    theme["colors"]["accent"] = json!(0);
    theme["colors"]["text"] = json!(255);
    theme["colors"]["customToken"] = json!("#123456");
    let validated = validate_theme_json("custom", &theme).expect("custom token");
    assert_eq!(
        validated.colors.get("customToken"),
        Some(&super::ThemeColorValue::Text("#123456".to_string()))
    );
}

#[test]
fn rejects_a_bad_variable_value_with_its_path() {
    let mut theme = dark_json();
    theme["vars"]["broken"] = json!({});
    let error = validate_theme_json("vars", &theme).expect_err("bad var");
    assert!(error.contains("/vars/broken: Expected a string or an integer 0-255"));
}

#[test]
fn parses_export_sections() {
    let mut theme = dark_json();
    theme["export"] = json!({
        "pageBg": "accent",
        "cardBg": 24,
        "infoBg": ""
    });
    let validated = validate_theme_json("export", &theme).expect("export");
    let export = validated.export.expect("export section");
    assert_eq!(
        export.page_bg,
        Some(super::ThemeColorValue::Text("accent".to_string()))
    );
    assert_eq!(export.card_bg, Some(super::ThemeColorValue::Index(24)));
    assert_eq!(
        export.info_bg,
        Some(super::ThemeColorValue::Text(String::new()))
    );
}

#[test]
fn rejects_a_mistyped_export_section_or_value() {
    let mut theme = dark_json();
    theme["export"] = json!([]);
    assert!(
        validate_theme_json("export", &theme)
            .expect_err("export array")
            .contains("/export: Expected object")
    );

    let mut theme = dark_json();
    theme["export"] = json!({ "pageBg": false });
    assert!(
        validate_theme_json("export", &theme)
            .expect_err("export value")
            .contains("/export/pageBg: Expected a string or an integer 0-255")
    );
}

#[test]
fn parse_content_strips_a_bom_and_reports_json_errors() {
    let with_bom = format!("\u{feff}{DARK}");
    parse_theme_json_content("dark.json", &with_bom).expect("bom tolerated");

    let error = parse_theme_json_content("bad.json", "{").expect_err("invalid json");
    assert!(error.starts_with("Failed to parse theme bad.json:"));
}
