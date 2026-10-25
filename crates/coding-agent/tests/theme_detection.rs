//! Port of `test/theme-detection.test.ts` at pin `60e7e76`.

#![expect(
    clippy::expect_used,
    reason = "the tests pin outcomes; an unexpected result panics the test by design"
)]
use pi_coding_agent::config::EnvLookup;
use pi_coding_agent::modes::interactive::theme::theme::{
    DetectionConfidence, DetectionSource, TerminalAutoThemeDetector,
    TerminalBackgroundThemeDetector, TerminalTheme, detect_terminal_background_from_env_with,
    detect_terminal_background_theme_with_env, detect_terminal_theme_for_auto_with_env,
    get_theme_for_rgb_color, parse_auto_theme_setting, resolve_theme_setting,
};
use pi_tui::terminal_colors::RgbColor;

fn env(entries: &[(&str, &str)]) -> EnvLookup {
    let owned: Vec<(String, String)> = entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect();
    Box::new(move |key| {
        owned
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    })
}

#[derive(Clone)]
struct Detector {
    background: Result<Option<RgbColor>, String>,
    scheme: Result<Option<TerminalTheme>, String>,
}

impl TerminalBackgroundThemeDetector for Detector {
    fn query_terminal_background_color(
        &self,
        _timeout_ms: u64,
    ) -> Result<Option<RgbColor>, String> {
        self.background.clone()
    }
}

impl TerminalAutoThemeDetector for Detector {
    fn query_terminal_color_scheme(
        &self,
        _timeout_ms: u64,
    ) -> Result<Option<TerminalTheme>, String> {
        self.scheme.clone()
    }
}

const fn detector(background: Result<Option<RgbColor>, String>) -> Detector {
    Detector {
        background,
        scheme: Ok(None),
    }
}

#[test]
fn detects_from_the_colorfgbg_background_index() {
    let light = detect_terminal_background_from_env_with(&env(&[("COLORFGBG", "0;15")]));
    assert_eq!(light.theme, TerminalTheme::Light);
    assert_eq!(light.source, DetectionSource::ColorFgBg);
    assert_eq!(light.confidence, DetectionConfidence::High);
    assert_eq!(light.detail, "background color index 15");

    let dark = detect_terminal_background_from_env_with(&env(&[("COLORFGBG", "15;0")]));
    assert_eq!(dark.theme, TerminalTheme::Dark);
}

#[test]
fn uses_the_last_colorfgbg_field() {
    assert_eq!(
        detect_terminal_background_from_env_with(&env(&[("COLORFGBG", "0;7;15")])).theme,
        TerminalTheme::Light
    );
    assert_eq!(
        detect_terminal_background_from_env_with(&env(&[("COLORFGBG", "0;15;bogus")])).theme,
        TerminalTheme::Light
    );
    assert_eq!(
        detect_terminal_background_from_env_with(&env(&[("COLORFGBG", "300;-1")])).theme,
        TerminalTheme::Dark
    );
}

#[test]
fn defaults_to_dark_without_hints() {
    let detection = detect_terminal_background_from_env_with(&env(&[]));
    assert_eq!(detection.theme, TerminalTheme::Dark);
    assert_eq!(detection.source, DetectionSource::Fallback);
    assert_eq!(detection.confidence, DetectionConfidence::Low);
    assert_eq!(detection.detail, "no terminal background hint found");
}

#[test]
fn prefers_the_terminal_background_query() {
    let detection = detect_terminal_background_theme_with_env(
        &detector(Ok(Some(RgbColor {
            r: 250,
            g: 250,
            b: 250,
        }))),
        250,
        &env(&[("COLORFGBG", "15;0")]),
    );
    assert_eq!(detection.theme, TerminalTheme::Light);
    assert_eq!(detection.source, DetectionSource::TerminalBackground);
    assert_eq!(detection.detail, "OSC 11 background rgb(250, 250, 250)");
}

#[test]
fn falls_back_to_the_environment_on_no_color_or_a_failed_query() {
    let no_color = detect_terminal_background_theme_with_env(
        &detector(Ok(None)),
        250,
        &env(&[("COLORFGBG", "15;0")]),
    );
    assert_eq!(no_color.source, DetectionSource::ColorFgBg);

    let failed = detect_terminal_background_theme_with_env(
        &detector(Err("terminal write failed".to_string())),
        250,
        &env(&[("COLORFGBG", "0;15")]),
    );
    assert_eq!(failed.theme, TerminalTheme::Light);
    assert_eq!(failed.source, DetectionSource::ColorFgBg);
}

#[test]
fn auto_detection_prefers_the_color_scheme_answer() {
    let detection = Detector {
        background: Ok(Some(RgbColor {
            r: 250,
            g: 250,
            b: 250,
        })),
        scheme: Ok(Some(TerminalTheme::Dark)),
    };
    assert_eq!(
        detect_terminal_theme_for_auto_with_env(&detection, 100, &env(&[])),
        TerminalTheme::Dark
    );
}

#[test]
fn auto_detection_falls_back_to_the_background_query() {
    let no_scheme = Detector {
        background: Ok(Some(RgbColor {
            r: 250,
            g: 250,
            b: 250,
        })),
        scheme: Ok(None),
    };
    assert_eq!(
        detect_terminal_theme_for_auto_with_env(&no_scheme, 100, &env(&[])),
        TerminalTheme::Light
    );

    let failed_scheme = Detector {
        background: Ok(None),
        scheme: Err("color-scheme query failed".to_string()),
    };
    assert_eq!(
        detect_terminal_theme_for_auto_with_env(
            &failed_scheme,
            100,
            &env(&[("COLORFGBG", "0;15")])
        ),
        TerminalTheme::Light
    );
}

#[test]
fn classifies_rgb_colors_by_luminance() {
    assert_eq!(
        get_theme_for_rgb_color(RgbColor { r: 8, g: 8, b: 8 }),
        TerminalTheme::Dark
    );
    assert_eq!(
        get_theme_for_rgb_color(RgbColor {
            r: 250,
            g: 250,
            b: 250
        }),
        TerminalTheme::Light
    );
}

#[test]
fn parses_and_resolves_automatic_theme_settings() {
    let auto = parse_auto_theme_setting(Some("light/dark")).expect("auto theme");
    assert_eq!(auto.light_theme, "light");
    assert_eq!(auto.dark_theme, "dark");
    assert_eq!(
        resolve_theme_setting(Some("dark"), TerminalTheme::Light),
        Some("dark".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("light/dark"), TerminalTheme::Light),
        Some("light".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("light/dark"), TerminalTheme::Dark),
        Some("dark".to_string())
    );
    assert_eq!(
        resolve_theme_setting(Some("light/dark/extra"), TerminalTheme::Dark),
        None
    );
}
