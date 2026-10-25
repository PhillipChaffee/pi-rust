//! Theme loading, colors, terminal detection, and the TUI theme getters,
//! upstream's `src/modes/interactive/theme/theme.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream keeps the active theme on `globalThis` behind a proxy so the tsx
//! and jiti module graphs share one instance. The port restates that as one
//! process-wide [`RwLock`] the [`theme`] accessor reads; every getter below
//! returns closures that resolve the current theme per call, so a theme
//! switch repaints mounted components without rebuilding them.
//!
//! `chalk` restates onto literal SGR sequences: the style helpers emit the
//! same open/reset pairs chalk does when color support is on. The built-in
//! themes compile in with `include_str!`, so the binary carries its palette.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError, RwLock};
use std::time::Duration;

use indexmap::IndexMap;
use pi_agent_core::types::ThinkingLevel;
use pi_tui::components::editor::EditorTheme;
use pi_tui::components::markdown::MarkdownTheme;
use pi_tui::components::select_list::SelectListTheme;
use pi_tui::components::settings_list::SettingsListTheme;
use pi_tui::terminal_colors::RgbColor;
use pi_tui::terminal_image::get_capabilities;

use super::theme_json::{ThemeColorValue, ValidatedThemeJson, parse_theme_json_content};
use crate::config::{EnvLookup, default_env_lookup, get_custom_themes_dir, get_themes_dir};
use crate::export_html::{ThemeColorsSource, ThemeExportColors};
use crate::source_info::SourceInfo;
use crate::utils::fs_watch::{WatchEvent, Watcher, close_watcher, watch_with_error_handler};
use crate::utils::syntax_highlight::{
    HighlightOptions, HighlightTheme, highlight, supports_language,
};

/// The color depth a built theme encodes at, upstream's `ColorMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    /// 24-bit color, upstream `"truecolor"`.
    Truecolor,
    /// The 256-color palette, upstream `"256color"`.
    Color256,
}

/// One constructed theme, upstream's `Theme`: ANSI escape strings per token.
#[derive(Clone)]
pub struct Theme {
    name: Option<String>,
    source_path: Option<String>,
    /// Provenance the resource loader records for registered themes,
    /// upstream's public `sourceInfo`.
    pub source_info: Option<SourceInfo>,
    fg_colors: IndexMap<String, String>,
    bg_colors: IndexMap<String, String>,
    mode: ColorMode,
}

impl std::fmt::Debug for Theme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Theme")
            .field("name", &self.name)
            .field("source_path", &self.source_path)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

/// Construction options, upstream's `{ name, sourcePath, sourceInfo }`.
#[derive(Clone, Debug, Default)]
pub struct ThemeOptions {
    /// The theme's display name, upstream `name`.
    pub name: Option<String>,
    /// The file the theme loaded from, upstream `sourcePath`.
    pub source_path: Option<String>,
    /// The resource provenance, upstream `sourceInfo`.
    pub source_info: Option<SourceInfo>,
}

/// The background tokens, upstream's `ThemeBg` split from the foreground
/// map.
const BG_COLOR_KEYS: &[&str] = &[
    "selectedBg",
    "searchMatchBg",
    "userMessageBg",
    "customMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
];

impl Theme {
    /// Build a theme, upstream's `new Theme(fgColors, bgColors, mode,
    /// options)`. Optional tokens fall back inside, as upstream's constructor
    /// does.
    ///
    /// # Panics
    /// A color value that is neither a hex string nor a 256-color index; the
    /// validator rejects those before construction.
    #[must_use]
    pub fn new(
        fg_colors: IndexMap<String, ThemeColorValue>,
        bg_colors: IndexMap<String, ThemeColorValue>,
        mode: ColorMode,
        options: ThemeOptions,
    ) -> Self {
        let mut fg_colors = fg_colors;
        let mut bg_colors = bg_colors;
        for (token, fallback) in [
            ("scrollbarTrack", "muted"),
            ("scrollbarThumb", "text"),
            ("thinkingMax", "thinkingXhigh"),
            ("searchMatchText", "text"),
        ] {
            apply_fallback(&mut fg_colors, token, fallback);
        }
        apply_fallback(&mut bg_colors, "searchMatchBg", "selectedBg");
        let fg_colors = build_color_map(&fg_colors, mode, true);
        let bg_colors = build_color_map(&bg_colors, mode, false);
        Self {
            name: options.name,
            source_path: options.source_path,
            source_info: options.source_info,
            fg_colors,
            bg_colors,
            mode,
        }
    }

    /// The theme's display name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The file the theme loaded from.
    #[must_use]
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// The color mode the theme resolved at.
    #[must_use]
    pub const fn color_mode(&self) -> ColorMode {
        self.mode
    }

    /// Decorate text with a foreground token, upstream's `fg`.
    ///
    /// # Panics
    /// An unknown token — a validated theme carries every required token and
    /// the constructor fills the optional ones, so this is a programming
    /// error, upstream's thrown `Unknown theme color`.
    #[must_use]
    #[expect(
        clippy::panic,
        reason = "upstream throws on an unknown token; the validated-theme invariant makes it unreachable"
    )]
    #[expect(
        clippy::option_if_let_else,
        reason = "the match keeps the panic arm inside the function, where the contract is documented, instead of a closure"
    )]
    pub fn fg(&self, color: &str, text: &str) -> String {
        match self.fg_colors.get(color) {
            Some(ansi) => format!("{ansi}{text}\x1b[39m"),
            None => panic!("Unknown theme color: {color}"),
        }
    }

    /// Decorate text with a background token, upstream's `bg`.
    ///
    /// # Panics
    /// An unknown token, as [`Theme::fg`].
    #[must_use]
    #[expect(
        clippy::panic,
        reason = "upstream throws on an unknown token; the validated-theme invariant makes it unreachable"
    )]
    #[expect(
        clippy::option_if_let_else,
        reason = "the match keeps the panic arm inside the function, where the contract is documented, instead of a closure"
    )]
    pub fn bg(&self, color: &str, text: &str) -> String {
        match self.bg_colors.get(color) {
            Some(ansi) => format!("{ansi}{text}\x1b[49m"),
            None => panic!("Unknown theme background color: {color}"),
        }
    }

    /// The raw foreground escape for a token, upstream's `getFgAnsi`.
    ///
    /// # Panics
    /// An unknown token, as [`Theme::fg`].
    #[must_use]
    #[expect(
        clippy::panic,
        reason = "upstream throws on an unknown token; the validated-theme invariant makes it unreachable"
    )]
    #[expect(
        clippy::option_if_let_else,
        reason = "the match keeps the panic arm inside the function, where the contract is documented, instead of a closure"
    )]
    pub fn get_fg_ansi(&self, color: &str) -> &str {
        match self.fg_colors.get(color) {
            Some(ansi) => ansi,
            None => panic!("Unknown theme color: {color}"),
        }
    }

    /// The raw background escape for a token, upstream's `getBgAnsi`.
    ///
    /// # Panics
    /// An unknown token, as [`Theme::fg`].
    #[must_use]
    #[expect(
        clippy::panic,
        reason = "upstream throws on an unknown token; the validated-theme invariant makes it unreachable"
    )]
    #[expect(
        clippy::option_if_let_else,
        reason = "the match keeps the panic arm inside the function, where the contract is documented, instead of a closure"
    )]
    pub fn get_bg_ansi(&self, color: &str) -> &str {
        match self.bg_colors.get(color) {
            Some(ansi) => ansi,
            None => panic!("Unknown theme background color: {color}"),
        }
    }
}

/// Apply a fallback token when the map omits it, upstream's `??` defaults.
fn apply_fallback(colors: &mut IndexMap<String, ThemeColorValue>, token: &str, fallback: &str) {
    if !colors.contains_key(token)
        && let Some(value) = colors.get(fallback)
    {
        let value = value.clone();
        colors.insert(token.to_string(), value);
    }
}

/// Encode every color value, upstream's per-map loop in the constructor.
fn build_color_map(
    colors: &IndexMap<String, ThemeColorValue>,
    mode: ColorMode,
    foreground: bool,
) -> IndexMap<String, String> {
    colors
        .iter()
        .map(|(key, value)| {
            let ansi = if foreground {
                fg_ansi(value, mode)
            } else {
                bg_ansi(value, mode)
            };
            (key.clone(), ansi)
        })
        .collect()
}

/// Encode a foreground value, upstream's `fgAnsi`.
///
/// # Panics
/// A text value that is neither empty nor a hex color — an unresolved
/// variable reference or a malformed hex string, upstream's thrown
/// `Invalid color value`.
#[expect(
    clippy::panic,
    reason = "upstream throws on an invalid color value; Theme::new documents the same contract"
)]
fn fg_ansi(color: &ThemeColorValue, mode: ColorMode) -> String {
    match color {
        ThemeColorValue::Text(text) if text.is_empty() => "\x1b[39m".to_string(),
        ThemeColorValue::Index(index) => format!("\x1b[38;5;{index}m"),
        ThemeColorValue::Text(text) if text.starts_with('#') => match mode {
            ColorMode::Truecolor => match hex_to_rgb(text) {
                Ok(rgb) => format!("\x1b[38;2;{};{};{}m", rgb.r, rgb.g, rgb.b),
                Err(error) => panic!("{error}"),
            },
            ColorMode::Color256 => match hex_to_256(text) {
                Ok(index) => format!("\x1b[38;5;{index}m"),
                Err(error) => panic!("{error}"),
            },
        },
        ThemeColorValue::Text(text) => panic!("Invalid color value: {text}"),
    }
}

/// Encode a background value, upstream's `bgAnsi`.
///
/// # Panics
/// As [`fg_ansi`].
#[expect(
    clippy::panic,
    reason = "upstream throws on an invalid color value; Theme::new documents the same contract"
)]
fn bg_ansi(color: &ThemeColorValue, mode: ColorMode) -> String {
    match color {
        ThemeColorValue::Text(text) if text.is_empty() => "\x1b[49m".to_string(),
        ThemeColorValue::Index(index) => format!("\x1b[48;5;{index}m"),
        ThemeColorValue::Text(text) if text.starts_with('#') => match mode {
            ColorMode::Truecolor => match hex_to_rgb(text) {
                Ok(rgb) => format!("\x1b[48;2;{};{};{}m", rgb.r, rgb.g, rgb.b),
                Err(error) => panic!("{error}"),
            },
            ColorMode::Color256 => match hex_to_256(text) {
                Ok(index) => format!("\x1b[48;5;{index}m"),
                Err(error) => panic!("{error}"),
            },
        },
        ThemeColorValue::Text(text) => panic!("Invalid color value: {text}"),
    }
}

/// Bold text, upstream's `chalk.bold`.
#[must_use]
pub fn bold(text: &str) -> String {
    format!("\x1b[1m{text}\x1b[22m")
}

/// Italic text, upstream's `chalk.italic`.
#[must_use]
pub fn italic(text: &str) -> String {
    format!("\x1b[3m{text}\x1b[23m")
}

/// Underlined text, upstream's `chalk.underline`.
#[must_use]
pub fn underline(text: &str) -> String {
    format!("\x1b[4m{text}\x1b[24m")
}

/// Inverse-video text, upstream's `chalk.inverse`.
#[must_use]
pub fn inverse(text: &str) -> String {
    format!("\x1b[7m{text}\x1b[27m")
}

/// Struck-through text, upstream's `chalk.strikethrough`.
#[must_use]
pub fn strikethrough(text: &str) -> String {
    format!("\x1b[9m{text}\x1b[29m")
}

/// Parse a `#rrggbb` color, upstream's `hexToRgb`.
///
/// # Errors
/// The string is not six hex digits after the `#`.
pub fn hex_to_rgb(hex: &str) -> Result<RgbColor, String> {
    let cleaned = hex.replace('#', "");
    if cleaned.len() != 6 {
        return Err(format!("Invalid hex color: {hex}"));
    }
    let channel = |range: std::ops::Range<usize>| u8::from_str_radix(&cleaned[range], 16).ok();
    match (channel(0..2), channel(2..4), channel(4..6)) {
        (Some(r), Some(g), Some(b)) => Ok(RgbColor { r, g, b }),
        _ => Err(format!("Invalid hex color: {hex}")),
    }
}

/// The 6x6x6 color-cube channel values, indices 0-5.
const CUBE_VALUES: [u8; 6] = [0, 95, 135, 175, 215, 255];

/// The closest cube channel index, upstream's `findClosestCubeIndex`.
fn closest_cube_index(value: u8) -> usize {
    let mut min_distance = u32::MAX;
    let mut min_index = 0;
    for (index, cube) in CUBE_VALUES.iter().enumerate() {
        let distance = u32::from(value).abs_diff(u32::from(*cube));
        if distance < min_distance {
            min_distance = distance;
            min_index = index;
        }
    }
    min_index
}

/// The closest grayscale-ramp index, upstream's `findClosestGrayIndex`: 24
/// values from 8 through 238, 10 apart.
fn closest_gray_index(gray: u8) -> u8 {
    let mut min_distance = u32::MAX;
    let mut min_index = 0;
    for index in 0_u8..24 {
        let value = 8 + index * 10;
        let distance = u32::from(gray).abs_diff(u32::from(value));
        if distance < min_distance {
            min_distance = distance;
            min_index = index;
        }
    }
    min_index
}

/// The weighted RGB distance, upstream's `colorDistance`.
#[expect(
    clippy::suboptimal_flops,
    reason = "restates upstream's `dr*dr*0.299 + dg*dg*0.587 + db*db*0.114` exactly"
)]
fn color_distance(rgb: RgbColor, other: RgbColor) -> f64 {
    let dr = f64::from(rgb.r) - f64::from(other.r);
    let dg = f64::from(rgb.g) - f64::from(other.g);
    let db = f64::from(rgb.b) - f64::from(other.b);
    dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114
}

/// The closest 256-color palette index, upstream's `rgbTo256`: the cube
/// unless a nearly neutral color is closer to the gray ramp.
#[must_use]
pub fn rgb_to_256(rgb: RgbColor) -> u8 {
    let r_index = closest_cube_index(rgb.r);
    let g_index = closest_cube_index(rgb.g);
    let b_index = closest_cube_index(rgb.b);
    let cube = RgbColor {
        r: CUBE_VALUES[r_index],
        g: CUBE_VALUES[g_index],
        b: CUBE_VALUES[b_index],
    };
    let cube_index = 16 + 36 * r_index + 6 * g_index + b_index;
    let cube_distance = color_distance(rgb, cube);

    #[expect(
        clippy::suboptimal_flops,
        reason = "restates upstream's `0.299*r + 0.587*g + 0.114*b` luma"
    )]
    let luma = 0.299 * f64::from(rgb.r) + 0.587 * f64::from(rgb.g) + 0.114 * f64::from(rgb.b);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "luma lies in 0..=255, so the rounded cast is exact"
    )]
    let gray = luma.round() as u8;
    let gray_index = closest_gray_index(gray);
    let gray_value = 8 + gray_index * 10;
    let gray_distance = color_distance(
        rgb,
        RgbColor {
            r: gray_value,
            g: gray_value,
            b: gray_value,
        },
    );

    let max = rgb.r.max(rgb.g).max(rgb.b);
    let min = rgb.r.min(rgb.g).min(rgb.b);
    if u16::from(max) - u16::from(min) < 10 && gray_distance < cube_distance {
        return 232 + gray_index;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "16 + 36*5 + 6*5 + 5 is 231, inside u8"
    )]
    let index = cube_index as u8;
    index
}

/// The 256-color index a hex color snaps to, upstream's `hexTo256`.
///
/// # Errors
/// The string is not a valid hex color.
pub fn hex_to_256(hex: &str) -> Result<u8, String> {
    Ok(rgb_to_256(hex_to_rgb(hex)?))
}

/// Convert a 256-color index to hex, upstream's `ansi256ToHex`.
#[must_use]
pub fn ansi256_to_hex(index: u8) -> String {
    const BASIC: [&str; 16] = [
        "#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0",
        "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff",
    ];
    if let Some(color) = BASIC.get(usize::from(index)) {
        return (*color).to_string();
    }
    if index < 232 {
        let cube_index = u32::from(index) - 16;
        let to_hex = |channel: u32| {
            if channel == 0 { 0 } else { 55 + channel * 40 }
        };
        return format!(
            "#{:02x}{:02x}{:02x}",
            to_hex(cube_index / 36),
            to_hex((cube_index % 36) / 6),
            to_hex(cube_index % 6)
        );
    }
    let gray = 8 + (u32::from(index) - 232) * 10;
    format!("#{gray:02x}{gray:02x}{gray:02x}")
}

/// Resolve a color value through the variable map, upstream's
/// `resolveVarRefs`.
///
/// # Errors
/// A circular reference or a name the variable map does not carry.
pub fn resolve_var_refs(
    value: &ThemeColorValue,
    vars: &IndexMap<String, ThemeColorValue>,
    visited: &mut BTreeSet<String>,
) -> Result<ThemeColorValue, String> {
    match value {
        ThemeColorValue::Index(_) => Ok(value.clone()),
        ThemeColorValue::Text(text) if text.is_empty() || text.starts_with('#') => {
            Ok(value.clone())
        }
        ThemeColorValue::Text(name) => {
            if visited.contains(name) {
                return Err(format!("Circular variable reference detected: {name}"));
            }
            let Some(next) = vars.get(name) else {
                return Err(format!("Variable reference not found: {name}"));
            };
            visited.insert(name.clone());
            resolve_var_refs(next, vars, visited)
        }
    }
}

/// Resolve every color in a map, upstream's `resolveThemeColors`.
///
/// # Errors
/// As [`resolve_var_refs`].
pub fn resolve_theme_colors(
    colors: &IndexMap<String, ThemeColorValue>,
    vars: &IndexMap<String, ThemeColorValue>,
) -> Result<IndexMap<String, ThemeColorValue>, String> {
    let mut resolved = IndexMap::new();
    for (key, value) in colors {
        let value = resolve_var_refs(value, vars, &mut BTreeSet::new())?;
        resolved.insert(key.clone(), value);
    }
    Ok(resolved)
}

/// Fill the optional tokens from their required fallbacks, upstream's
/// `withThemeColorFallbacks`.
fn with_theme_color_fallbacks(
    colors: &IndexMap<String, ThemeColorValue>,
) -> IndexMap<String, ThemeColorValue> {
    let mut with_fallbacks = colors.clone();
    for (token, fallback) in [
        ("scrollbarTrack", "muted"),
        ("scrollbarThumb", "text"),
        ("thinkingMax", "thinkingXhigh"),
        ("searchMatchBg", "selectedBg"),
        ("searchMatchText", "text"),
    ] {
        apply_fallback(&mut with_fallbacks, token, fallback);
    }
    with_fallbacks
}

/// Build a theme from a validated document, upstream's `createTheme`.
///
/// # Errors
/// A variable reference that does not resolve.
pub fn create_theme(
    theme_json: &ValidatedThemeJson,
    mode: Option<ColorMode>,
    source_path: Option<String>,
) -> Result<Theme, String> {
    let color_mode = mode.unwrap_or_else(|| {
        if get_capabilities().true_color {
            ColorMode::Truecolor
        } else {
            ColorMode::Color256
        }
    });
    let resolved = resolve_theme_colors(
        &with_theme_color_fallbacks(&theme_json.colors),
        &theme_json.vars,
    )?;
    let mut fg_colors = IndexMap::new();
    let mut bg_colors = IndexMap::new();
    for (key, value) in resolved {
        if BG_COLOR_KEYS.contains(&key.as_str()) {
            bg_colors.insert(key, value);
        } else {
            fg_colors.insert(key, value);
        }
    }
    Ok(Theme::new(
        fg_colors,
        bg_colors,
        color_mode,
        ThemeOptions {
            name: Some(theme_json.name.clone()),
            source_path,
            source_info: None,
        },
    ))
}

/// One theme the picker lists, upstream's `ThemeInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThemeInfo {
    /// The display name.
    pub name: String,
    /// The file the theme loads from; `None` for an in-memory registration.
    pub path: Option<String>,
}

/// The built-in theme documents, parsed once from the compiled-in assets.
///
/// # Panics
/// A compiled-in asset fails validation — a build-time invariant, since the
/// same files ship as upstream's built-ins.
fn builtin_themes() -> &'static IndexMap<String, ValidatedThemeJson> {
    static BUILTIN_THEMES: LazyLock<IndexMap<String, ValidatedThemeJson>> = LazyLock::new(|| {
        let mut themes = IndexMap::new();
        for (name, content) in [
            ("dark", include_str!("dark.json")),
            ("light", include_str!("light.json")),
        ] {
            match parse_theme_json_content(name, content) {
                Ok(theme) => {
                    themes.insert(name.to_string(), theme);
                }
                Err(error) => {
                    #[expect(
                        clippy::panic,
                        reason = "the built-in JSON is validated by the crate's own tests; a broken asset is a build defect"
                    )]
                    {
                        panic!("built-in theme {name} is invalid: {error}");
                    }
                }
            }
        }
        themes
    });
    &BUILTIN_THEMES
}

/// The available theme names, upstream's `getAvailableThemes`.
#[must_use]
pub fn get_available_themes() -> Vec<String> {
    get_available_themes_with_paths()
        .into_iter()
        .map(|info| info.name)
        .collect()
}

/// The available themes with their paths, upstream's
/// `getAvailableThemesWithPaths`: built-ins, custom files by their content
/// name, then registered themes.
#[must_use]
pub fn get_available_themes_with_paths() -> Vec<ThemeInfo> {
    let (themes_dir, custom_themes_dir) = theme_dirs();
    get_available_themes_with_paths_with(&themes_dir, &custom_themes_dir)
}

/// [`get_available_themes_with_paths`] over injected directories, the test
/// seam for [`PI_CODING_AGENT_DIR`](crate::config::ENV_AGENT_DIR).
#[must_use]
pub fn get_available_themes_with_paths_with(
    themes_dir: &Path,
    custom_themes_dir: &Path,
) -> Vec<ThemeInfo> {
    let mut result: Vec<ThemeInfo> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut add = |info: ThemeInfo| {
        if seen.insert(info.name.clone()) {
            result.push(info);
        }
    };

    for name in builtin_themes().keys() {
        add(ThemeInfo {
            name: name.clone(),
            path: Some(
                themes_dir
                    .join(format!("{name}.json"))
                    .to_string_lossy()
                    .into_owned(),
            ),
        });
    }

    for info in custom_theme_infos(custom_themes_dir) {
        add(info);
    }

    if let Ok(registered) = REGISTERED_THEMES.lock() {
        for (name, theme) in registered.iter() {
            add(ThemeInfo {
                name: name.clone(),
                path: theme.source_path().map(str::to_string),
            });
        }
    }

    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

/// The custom theme files, upstream's `getCustomThemeInfos`; a file that
/// fails to load is skipped, the resource loader's diagnostics own it.
fn custom_theme_infos(custom_themes_dir: &Path) -> Vec<ThemeInfo> {
    let Ok(entries) = fs::read_dir(custom_themes_dir) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
            continue;
        }
        if let Ok(theme) = load_theme_from_path(&path, None)
            && let Some(name) = theme.name()
        {
            result.push(ThemeInfo {
                name: name.to_string(),
                path: Some(path.to_string_lossy().into_owned()),
            });
        }
    }
    result
}

/// Reject a name carrying the automatic-setting separator, upstream's
/// `assertThemeNameIsValid`.
///
/// # Errors
/// The name contains `/`.
pub fn assert_theme_name_is_valid(name: &str) -> Result<(), String> {
    if name.contains('/') {
        return Err(format!(
            "Invalid theme name \"{name}\": theme names cannot contain \"/\" because it is reserved for automatic light/dark theme settings."
        ));
    }
    Ok(())
}

/// Read one theme document by name, upstream's `loadThemeJson`.
///
/// # Errors
/// The theme is unknown, its file does not read, or it fails validation.
pub fn load_theme_json(name: &str) -> Result<ValidatedThemeJson, String> {
    let (_, custom_themes_dir) = theme_dirs();
    load_theme_json_with(name, &custom_themes_dir)
}

/// [`load_theme_json`] over an injected custom-theme directory, the test
/// seam for [`PI_CODING_AGENT_DIR`](crate::config::ENV_AGENT_DIR).
///
/// # Errors
/// As [`load_theme_json`].
pub fn load_theme_json_with(
    name: &str,
    custom_themes_dir: &Path,
) -> Result<ValidatedThemeJson, String> {
    if let Some(theme) = builtin_themes().get(name) {
        return Ok(theme.clone());
    }
    if let Ok(registered) = REGISTERED_THEMES.lock()
        && let Some(theme) = registered.get(name)
    {
        let Some(source_path) = theme.source_path() else {
            return Err(format!(
                "Theme \"{name}\" does not have a source path for export"
            ));
        };
        let content = fs::read_to_string(source_path)
            .map_err(|error| format!("Failed to read theme {source_path}: {error}"))?;
        return parse_theme_json_content(source_path, &content);
    }
    let theme_path = custom_themes_dir.join(format!("{name}.json"));
    if !theme_path.exists() {
        return Err(format!("Theme not found: {name}"));
    }
    let content = fs::read_to_string(&theme_path)
        .map_err(|error| format!("Failed to read theme {}: {error}", theme_path.display()))?;
    parse_theme_json_content(name, &content)
}

/// Load a theme from a file, upstream's `loadThemeFromPath`.
///
/// # Errors
/// The file does not read or its document fails validation.
pub fn load_theme_from_path(theme_path: &Path, mode: Option<ColorMode>) -> Result<Theme, String> {
    let label = theme_path.to_string_lossy();
    let content = fs::read_to_string(theme_path)
        .map_err(|error| format!("Failed to read theme {label}: {error}"))?;
    let theme_json = parse_theme_json_content(&label, &content)?;
    create_theme(&theme_json, mode, Some(label.into_owned()))
}

/// Load a theme by name, upstream's `loadTheme`: a registered instance wins
/// over the on-disk lookup.
///
/// # Errors
/// As [`load_theme_json`], plus variable resolution.
pub fn load_theme(name: &str, mode: Option<ColorMode>) -> Result<Theme, String> {
    if let Ok(registered) = REGISTERED_THEMES.lock()
        && let Some(theme) = registered.get(name)
    {
        return Ok((**theme).clone());
    }
    let theme_json = load_theme_json(name)?;
    create_theme(&theme_json, mode, None)
}

/// Load a theme by name, answering `None` on any failure, upstream's
/// `getThemeByName`.
#[must_use]
pub fn get_theme_by_name(name: &str) -> Option<Theme> {
    load_theme(name, None).ok()
}

/// The terminal's appearance, upstream's `TerminalTheme`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalTheme {
    /// A dark terminal background.
    Dark,
    /// A light terminal background.
    Light,
}

impl TerminalTheme {
    /// The theme name the built-in palettes carry for this appearance.
    #[must_use]
    pub const fn theme_name(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }
}

/// A parsed automatic theme setting, upstream's `{ lightTheme, darkTheme }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AutoTheme {
    /// The theme for a light terminal, the left side of `light/dark`.
    pub light_theme: String,
    /// The theme for a dark terminal, the right side.
    pub dark_theme: String,
}

/// Parse `light/dark`, upstream's `parseAutoThemeSetting`.
#[must_use]
pub fn parse_auto_theme_setting(theme_setting: Option<&str>) -> Option<AutoTheme> {
    let theme_setting = theme_setting?;
    let mut parts = theme_setting.split('/');
    let light_theme = parts.next()?.trim();
    let dark_theme = parts.next()?.trim();
    if parts.next().is_some() || light_theme.is_empty() || dark_theme.is_empty() {
        return None;
    }
    Some(AutoTheme {
        light_theme: light_theme.to_string(),
        dark_theme: dark_theme.to_string(),
    })
}

/// Resolve a setting against the terminal appearance, upstream's
/// `resolveThemeSetting`.
#[must_use]
pub fn resolve_theme_setting(
    theme_setting: Option<&str>,
    terminal_theme: TerminalTheme,
) -> Option<String> {
    if let Some(auto_theme) = parse_auto_theme_setting(theme_setting) {
        return Some(match terminal_theme {
            TerminalTheme::Light => auto_theme.light_theme,
            TerminalTheme::Dark => auto_theme.dark_theme,
        });
    }
    let theme_setting = theme_setting?;
    if theme_setting.contains('/') {
        return None;
    }
    Some(theme_setting.to_string())
}

/// Where a detection answer came from, upstream's `source`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionSource {
    /// The OSC 11 terminal query.
    TerminalBackground,
    /// The `COLORFGBG` environment variable.
    ColorFgBg,
    /// No hint was available.
    Fallback,
}

/// How strongly a detection answer is trusted, upstream's `confidence`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionConfidence {
    /// The terminal or `COLORFGBG` answered.
    High,
    /// The fallback answered.
    Low,
}

/// One terminal-appearance detection, upstream's `TerminalThemeDetection`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerminalThemeDetection {
    /// The detected appearance.
    pub theme: TerminalTheme,
    /// Where it came from.
    pub source: DetectionSource,
    /// The human-readable detail, upstream `detail`.
    pub detail: String,
    /// How strongly it is trusted.
    pub confidence: DetectionConfidence,
}

/// Query seam for terminal appearance, upstream's
/// `TerminalBackgroundThemeDetector`; the blocking port of its promise.
pub trait TerminalBackgroundThemeDetector {
    /// Query the terminal's default background with OSC 11.
    ///
    /// # Errors
    /// The query could not be started or completed; the caller falls back to
    /// the environment hints.
    fn query_terminal_background_color(&self, timeout_ms: u64) -> Result<Option<RgbColor>, String>;
}

/// The auto-theme detector, upstream's `TerminalAutoThemeDetector`: the
/// color-scheme query is optional upstream and required here.
pub trait TerminalAutoThemeDetector: TerminalBackgroundThemeDetector {
    /// Query the terminal's light/dark preference (`CSI ? 996 n`).
    ///
    /// # Errors
    /// As [`TerminalBackgroundThemeDetector::query_terminal_background_color`].
    fn query_terminal_color_scheme(&self, timeout_ms: u64)
    -> Result<Option<TerminalTheme>, String>;
}

/// The background color index `COLORFGBG` carries, upstream's
/// `getColorFgBgBackgroundIndex`: the last field in 0-255.
fn color_fgbg_background_index(colorfgbg: &str) -> Option<u8> {
    colorfgbg
        .split(';')
        .rev()
        .filter_map(|part| part.trim().parse::<i64>().ok())
        .find(|index| (0..=255).contains(index))
        .and_then(|index| u8::try_from(index).ok())
}

/// Relative luminance, upstream's `getRgbColorLuminance`.
fn rgb_color_luminance(rgb: RgbColor) -> f64 {
    let to_linear = |channel: u8| {
        let value = f64::from(channel) / 255.0;
        if value <= 0.039_28 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    0.0722f64.mul_add(
        to_linear(rgb.b),
        0.7152f64.mul_add(to_linear(rgb.g), 0.2126 * to_linear(rgb.r)),
    )
}

/// Luminance of a 256-color index, upstream's `getAnsiColorLuminance`.
fn ansi_color_luminance(index: u8) -> f64 {
    hex_to_rgb(&ansi256_to_hex(index)).map_or(0.0, rgb_color_luminance)
}

/// Classify an RGB color, upstream's `getThemeForRgbColor`.
#[must_use]
pub fn get_theme_for_rgb_color(rgb: RgbColor) -> TerminalTheme {
    if rgb_color_luminance(rgb) >= 0.5 {
        TerminalTheme::Light
    } else {
        TerminalTheme::Dark
    }
}

/// Detect the terminal appearance from the environment, upstream's
/// `detectTerminalBackgroundFromEnv`.
#[must_use]
pub fn detect_terminal_background_from_env() -> TerminalThemeDetection {
    detect_terminal_background_from_env_with(&default_env_lookup())
}

/// [`detect_terminal_background_from_env`] over an injected environment.
#[must_use]
pub fn detect_terminal_background_from_env_with(env: &EnvLookup) -> TerminalThemeDetection {
    let colorfgbg = env("COLORFGBG").unwrap_or_default();
    if let Some(index) = color_fgbg_background_index(&colorfgbg) {
        return TerminalThemeDetection {
            theme: if ansi_color_luminance(index) >= 0.5 {
                TerminalTheme::Light
            } else {
                TerminalTheme::Dark
            },
            source: DetectionSource::ColorFgBg,
            detail: format!("background color index {index}"),
            confidence: DetectionConfidence::High,
        };
    }
    TerminalThemeDetection {
        theme: TerminalTheme::Dark,
        source: DetectionSource::Fallback,
        detail: "no terminal background hint found".to_string(),
        confidence: DetectionConfidence::Low,
    }
}

/// Detect the terminal appearance through OSC 11, falling back to the
/// environment hints, upstream's `detectTerminalBackgroundTheme`.
#[must_use]
pub fn detect_terminal_background_theme(
    ui: &dyn TerminalBackgroundThemeDetector,
    timeout_ms: u64,
) -> TerminalThemeDetection {
    detect_terminal_background_theme_with_env(ui, timeout_ms, &default_env_lookup())
}

/// [`detect_terminal_background_theme`] over an injected environment.
#[must_use]
pub fn detect_terminal_background_theme_with_env(
    ui: &dyn TerminalBackgroundThemeDetector,
    timeout_ms: u64,
    env: &EnvLookup,
) -> TerminalThemeDetection {
    if let Ok(Some(rgb)) = ui.query_terminal_background_color(timeout_ms) {
        return TerminalThemeDetection {
            theme: get_theme_for_rgb_color(rgb),
            source: DetectionSource::TerminalBackground,
            detail: format!("OSC 11 background rgb({}, {}, {})", rgb.r, rgb.g, rgb.b),
            confidence: DetectionConfidence::High,
        };
    }
    detect_terminal_background_from_env_with(env)
}

/// Detect the appearance for an automatic theme, upstream's
/// `detectTerminalThemeForAuto`: the color-scheme answer wins, the OSC 11
/// answer is the fallback.
///
/// Upstream starts both queries concurrently; the blocking port asks the
/// color-scheme query first and only then the background query, so a slow
/// color-scheme answer costs its timeout before the fallback starts.
#[must_use]
pub fn detect_terminal_theme_for_auto(
    ui: &dyn TerminalAutoThemeDetector,
    timeout_ms: u64,
) -> TerminalTheme {
    detect_terminal_theme_for_auto_with_env(ui, timeout_ms, &default_env_lookup())
}

/// [`detect_terminal_theme_for_auto`] over an injected environment.
#[must_use]
pub fn detect_terminal_theme_for_auto_with_env(
    ui: &dyn TerminalAutoThemeDetector,
    timeout_ms: u64,
    env: &EnvLookup,
) -> TerminalTheme {
    if let Ok(Some(theme)) = ui.query_terminal_color_scheme(timeout_ms) {
        return theme;
    }
    detect_terminal_background_theme_with_env(ui, timeout_ms, env).theme
}

/// The appearance a fresh session starts under, upstream's
/// `getDefaultTheme`.
#[must_use]
pub fn get_default_theme() -> TerminalTheme {
    detect_terminal_background_from_env().theme
}

/// The outcome of a theme switch, upstream's `{ success, error? }`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThemeResult {
    /// Whether the theme loaded.
    pub success: bool,
    /// The failure message when it did not.
    pub error: Option<String>,
}

impl ThemeResult {
    /// A successful switch.
    #[must_use]
    pub const fn success() -> Self {
        Self {
            success: true,
            error: None,
        }
    }

    /// A failed switch carrying the loader's message.
    #[must_use]
    pub const fn failure(error: String) -> Self {
        Self {
            success: false,
            error: Some(error),
        }
    }
}

/// The active theme, `None` before [`init_theme`] runs.
static GLOBAL_THEME: LazyLock<RwLock<Option<Arc<Theme>>>> = LazyLock::new(|| RwLock::new(None));

/// The active theme's requested name, upstream `currentThemeName`.
static CURRENT_THEME_NAME: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// The registered-theme map, upstream `registeredThemes`.
type RegisteredThemes = Mutex<IndexMap<String, Arc<Theme>>>;

/// Themes registered by the resource loader, upstream `registeredThemes`.
static REGISTERED_THEMES: LazyLock<RegisteredThemes> =
    LazyLock::new(|| Mutex::new(IndexMap::new()));

/// The live custom-theme watch, upstream `themeWatcher`.
static THEME_WATCHER: LazyLock<Mutex<Option<Watcher>>> = LazyLock::new(|| Mutex::new(None));

/// The reload debounce generation; a bump invalidates pending reloads.
static THEME_RELOAD_GENERATION: AtomicU64 = AtomicU64::new(0);

/// The theme-change callback, upstream `onThemeChangeCallback`.
type ThemeChangeCallback = Arc<dyn Fn() + Send + Sync>;

/// The one registered change callback, upstream `onThemeChangeCallback`.
static ON_THEME_CHANGE: LazyLock<Mutex<Option<ThemeChangeCallback>>> =
    LazyLock::new(|| Mutex::new(None));

/// A test override for the built-in and custom theme directories, the port's
/// stand-in for upstream's `PI_CODING_AGENT_DIR` stubs.
static THEME_DIRS_OVERRIDE: LazyLock<RwLock<Option<(PathBuf, PathBuf)>>> =
    LazyLock::new(|| RwLock::new(None));

/// Redirect the theme disk lookups, or `None` to restore the process paths.
///
/// Upstream's tests stub `PI_CODING_AGENT_DIR`; the port cannot write the
/// process environment under `forbid(unsafe_code)`, so the module's own
/// directory pair is the injectable seam.
pub fn set_theme_dirs_override(dirs: Option<(PathBuf, PathBuf)>) {
    *THEME_DIRS_OVERRIDE
        .write()
        .unwrap_or_else(PoisonError::into_inner) = dirs;
}

/// The directory pair disk lookups use: the override, else the package's
/// built-in directory and the agent's custom one.
fn theme_dirs() -> (PathBuf, PathBuf) {
    THEME_DIRS_OVERRIDE
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
        .unwrap_or_else(|| (get_themes_dir(), get_custom_themes_dir()))
}

/// Lock a mutex, reading through poisoning, the house's no-panic recovery.
fn lock_or_recover<T>(lock: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Read the active theme, upstream's `theme` proxy.
///
/// # Panics
/// Before [`init_theme`] runs, upstream's thrown
/// `Theme not initialized. Call initTheme() first.`
#[must_use]
#[expect(
    clippy::panic,
    reason = "upstream's proxy throws the same error; every consumer runs after init_theme"
)]
#[expect(
    clippy::option_if_let_else,
    reason = "the match keeps the panic arm inside the function instead of a closure"
)]
pub fn theme() -> Arc<Theme> {
    let active = GLOBAL_THEME
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    match active {
        Some(theme) => theme,
        None => panic!("Theme not initialized. Call init_theme() first."),
    }
}

/// Install the process-wide theme, upstream's `setGlobalTheme`.
fn set_global_theme(theme: Arc<Theme>) {
    *GLOBAL_THEME.write().unwrap_or_else(PoisonError::into_inner) = Some(theme);
}

/// The active theme's requested name, upstream `currentThemeName`.
#[must_use]
pub fn current_theme_name() -> Option<String> {
    lock_or_recover(&CURRENT_THEME_NAME).clone()
}

/// Set the requested name, upstream's direct assignment.
fn set_current_theme_name(name: Option<String>) {
    *lock_or_recover(&CURRENT_THEME_NAME) = name;
}

/// Replace the registered theme set, upstream's `setRegisteredThemes`.
///
/// # Errors
/// A theme name carries `/`, which the automatic light/dark settings reserve.
pub fn set_registered_themes(themes: Vec<Arc<Theme>>) -> Result<(), String> {
    let mut registered = IndexMap::new();
    for theme in themes {
        if let Some(name) = theme.name() {
            assert_theme_name_is_valid(name)?;
            registered.insert(name.to_string(), theme);
        }
    }
    *lock_or_recover(&REGISTERED_THEMES) = registered;
    Ok(())
}

/// Register the theme-change callback, upstream's `onThemeChange`: one
/// callback, a later registration replaces it.
pub fn on_theme_change(callback: ThemeChangeCallback) {
    *lock_or_recover(&ON_THEME_CHANGE) = Some(callback);
}

/// Invoke the registered change callback, upstream's guarded call site.
fn notify_theme_change() {
    let callback = lock_or_recover(&ON_THEME_CHANGE).clone();
    if let Some(callback) = callback {
        callback();
    }
}

/// Load and install a theme by name, falling back to `dark`, upstream's
/// `initTheme`.
///
/// # Panics
/// The compiled-in `dark` theme fails to load — a build defect.
pub fn init_theme(theme_name: Option<&str>, enable_watcher: bool) {
    let name = theme_name.map_or_else(
        || get_default_theme().theme_name().to_string(),
        str::to_string,
    );
    set_current_theme_name(Some(name.clone()));
    if let Ok(theme) = load_theme(&name, None) {
        set_global_theme(Arc::new(theme));
        if enable_watcher {
            start_theme_watcher();
        }
        return;
    }
    set_current_theme_name(Some("dark".to_string()));
    match load_theme("dark", None) {
        Ok(theme) => set_global_theme(Arc::new(theme)),
        Err(error) => {
            #[expect(
                clippy::panic,
                reason = "the compiled-in fallback theme is a build-time invariant"
            )]
            {
                panic!("the built-in dark theme failed to load: {error}");
            }
        }
    }
}

/// Load and install a theme by name, answering the failure instead of
/// falling back silently, upstream's `setTheme`.
#[must_use]
pub fn set_theme(name: &str, enable_watcher: bool) -> ThemeResult {
    set_current_theme_name(Some(name.to_string()));
    match load_theme(name, None) {
        Ok(theme) => {
            set_global_theme(Arc::new(theme));
            if enable_watcher {
                start_theme_watcher();
            }
            notify_theme_change();
            ThemeResult::success()
        }
        Err(error) => {
            set_current_theme_name(Some("dark".to_string()));
            if let Ok(theme) = load_theme("dark", None) {
                set_global_theme(Arc::new(theme));
            }
            ThemeResult::failure(error)
        }
    }
}

/// Install a constructed theme directly, upstream's `setThemeInstance`: a
/// direct instance has nothing to watch.
pub fn set_theme_instance(theme_instance: Arc<Theme>) {
    set_global_theme(theme_instance);
    set_current_theme_name(Some("<in-memory>".to_string()));
    stop_theme_watcher();
    notify_theme_change();
}

/// Watch the active custom theme file, upstream's `startThemeWatcher`:
/// changes debounce 100 ms, then reload in place.
fn start_theme_watcher() {
    stop_theme_watcher();
    let Some(watched_name) = current_theme_name() else {
        return;
    };
    if watched_name == "dark" || watched_name == "light" {
        return;
    }
    let custom_themes_dir = theme_dirs().1;
    let theme_file = custom_themes_dir.join(format!("{watched_name}.json"));
    if !theme_file.exists() {
        return;
    }
    let watched_file_name = format!("{watched_name}.json");
    let listener_name = watched_name;
    let listener_dir = custom_themes_dir.to_string_lossy().into_owned();
    let listener = move |event: WatchEvent| {
        if current_theme_name().as_deref() != Some(listener_name.as_str()) {
            return;
        }
        if let Some(filename) = event.filename
            && filename != watched_file_name
        {
            return;
        }
        schedule_theme_reload(&listener_name, &theme_file);
    };
    let watcher = watch_with_error_handler(&listener_dir, listener, drop_theme_watcher);
    *lock_or_recover(&THEME_WATCHER) = watcher;
}

/// Debounce and apply one theme-file change, upstream's `scheduleReload`.
///
/// A stale generation — another edit, a switch, or a stop — drops the
/// reload, so the last event of a burst wins.
fn schedule_theme_reload(watched_name: &str, theme_file: &Path) {
    let generation = THEME_RELOAD_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    let watched_name = watched_name.to_string();
    let theme_file = theme_file.to_path_buf();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        if THEME_RELOAD_GENERATION.load(Ordering::SeqCst) != generation {
            return;
        }
        if current_theme_name().as_deref() != Some(watched_name.as_str()) {
            return;
        }
        // A save-in-progress can leave the file briefly missing or invalid;
        // the last good theme stays active, as upstream keeps it.
        if !theme_file.exists() {
            return;
        }
        let Ok(reloaded) = load_theme_from_path(&theme_file, None) else {
            return;
        };
        let reloaded = Arc::new(reloaded);
        lock_or_recover(&REGISTERED_THEMES).insert(watched_name, Arc::clone(&reloaded));
        set_global_theme(reloaded);
        notify_theme_change();
    });
}

/// Drop the watch after a runtime watch error, upstream's close-on-error
/// callback.
fn drop_theme_watcher() {
    let watcher = lock_or_recover(&THEME_WATCHER).take();
    close_watcher(watcher);
}

/// Stop the theme watch and invalidate a pending reload, upstream's
/// `stopThemeWatcher`.
pub fn stop_theme_watcher() {
    THEME_RELOAD_GENERATION.fetch_add(1, Ordering::SeqCst);
    let watcher = lock_or_recover(&THEME_WATCHER).take();
    close_watcher(watcher);
}

/// Resolve a theme's colors as CSS-compatible hex strings, upstream's
/// `getResolvedThemeColors`.
///
/// # Errors
/// The named theme does not load or its variables do not resolve.
pub fn get_resolved_theme_colors(
    theme_name: Option<&str>,
) -> Result<BTreeMap<String, String>, String> {
    let name = theme_name
        .map(str::to_string)
        .or_else(current_theme_name)
        .unwrap_or_else(|| get_default_theme().theme_name().to_string());
    let theme_json = load_theme_json(&name)?;
    let resolved = resolve_theme_colors(
        &with_theme_color_fallbacks(&theme_json.colors),
        &theme_json.vars,
    )?;
    let default_text = if is_light_theme(Some(&name)) {
        "#000000"
    } else {
        "#e5e5e7"
    };
    let mut css_colors = BTreeMap::new();
    for (key, value) in resolved {
        let color = match value {
            ThemeColorValue::Index(index) => ansi256_to_hex(index),
            ThemeColorValue::Text(text) if text.is_empty() => default_text.to_string(),
            ThemeColorValue::Text(text) => text,
        };
        css_colors.insert(key, color);
    }
    Ok(css_colors)
}

/// Whether a theme is a light theme, upstream's `isLightTheme`; the name is
/// the whole signal.
#[must_use]
pub fn is_light_theme(theme_name: Option<&str>) -> bool {
    theme_name == Some("light")
}

/// The explicit export colors a theme carries, upstream's
/// `getThemeExportColors`; a theme that fails to load contributes nothing.
#[must_use]
pub fn get_theme_export_colors(theme_name: Option<&str>) -> ThemeExportColors {
    let (_, custom_themes_dir) = theme_dirs();
    get_theme_export_colors_with(theme_name, &custom_themes_dir)
}

/// [`get_theme_export_colors`] over an injected custom-theme directory, the
/// test seam.
#[must_use]
pub fn get_theme_export_colors_with(
    theme_name: Option<&str>,
    custom_themes_dir: &Path,
) -> ThemeExportColors {
    let name = theme_name
        .map(str::to_string)
        .or_else(current_theme_name)
        .unwrap_or_else(|| get_default_theme().theme_name().to_string());
    let Ok(theme_json) = load_theme_json_with(&name, custom_themes_dir) else {
        return ThemeExportColors::default();
    };
    let Some(export) = theme_json.export else {
        return ThemeExportColors::default();
    };
    let resolve = |value: Option<&ThemeColorValue>| -> Option<String> {
        let value = value?;
        match resolve_var_refs(value, &theme_json.vars, &mut BTreeSet::new()) {
            Ok(ThemeColorValue::Index(index)) => Some(ansi256_to_hex(index)),
            Ok(ThemeColorValue::Text(text)) if text.is_empty() => None,
            Ok(ThemeColorValue::Text(text)) => Some(text),
            Err(_) => None,
        }
    };
    ThemeExportColors {
        page_bg: resolve(export.page_bg.as_ref()),
        card_bg: resolve(export.card_bg.as_ref()),
        info_bg: resolve(export.info_bg.as_ref()),
    }
}

/// The theme-backed source for the HTML export, upstream's module-level
/// `getResolvedThemeColors`/`getThemeExportColors` reads.
///
/// The export trait cannot report failures, so a theme that fails to load
/// contributes the empty map, the same colorless fallback upstream's export
/// takes when the read throws.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThemeSystemSource;

impl ThemeColorsSource for ThemeSystemSource {
    fn resolved_colors(&self, theme_name: Option<&str>) -> BTreeMap<String, String> {
        get_resolved_theme_colors(theme_name).unwrap_or_default()
    }

    fn export_colors(&self, theme_name: Option<&str>) -> ThemeExportColors {
        get_theme_export_colors(theme_name)
    }
}

/// The highlight scope map, upstream's `CliHighlightTheme`: hljs vocabulary
/// to theme colors.
fn cli_highlight_theme() -> HighlightTheme<'static> {
    type Formatter = Arc<dyn Fn(&str) -> String>;
    let mut map: std::collections::HashMap<&'static str, Formatter> =
        std::collections::HashMap::new();
    for (scope, token) in [
        ("keyword", "syntaxKeyword"),
        ("built_in", "syntaxType"),
        ("literal", "syntaxNumber"),
        ("number", "syntaxNumber"),
        ("regexp", "syntaxString"),
        ("string", "syntaxString"),
        ("comment", "syntaxComment"),
        ("doctag", "syntaxComment"),
        ("meta", "muted"),
        ("function", "syntaxFunction"),
        ("title", "syntaxFunction"),
        ("class", "syntaxType"),
        ("type", "syntaxType"),
        ("tag", "syntaxPunctuation"),
        ("name", "syntaxKeyword"),
        ("attr", "syntaxVariable"),
        ("variable", "syntaxVariable"),
        ("params", "syntaxVariable"),
        ("operator", "syntaxOperator"),
        ("punctuation", "syntaxPunctuation"),
        ("addition", "toolDiffAdded"),
        ("deletion", "toolDiffRemoved"),
    ] {
        let formatter: Formatter = Arc::new(move |text: &str| theme().fg(token, text));
        map.insert(scope, formatter);
    }
    let emphasis: Formatter = Arc::new(|text: &str| italic(text));
    map.insert("emphasis", emphasis);
    let strong: Formatter = Arc::new(|text: &str| bold(text));
    map.insert("strong", strong);
    let link: Formatter = Arc::new(|text: &str| underline(text));
    map.insert("link", link);
    HighlightTheme::new(map)
}

/// Highlight code by language, upstream's `highlightCode`.
///
/// Unknown languages and empty language hints skip highlighting: upstream
/// records that highlight.js's auto-detection misreads prose as `AppleScript`
/// and `LiveCodeServer`, so only an explicit, supported language highlights.
#[must_use]
pub fn highlight_code(code: &str, lang: Option<&str>) -> Vec<String> {
    let Some(lang) = lang.filter(|lang| supports_language(lang)) else {
        return code
            .split('\n')
            .map(|line| theme().fg("mdCodeBlock", line))
            .collect();
    };
    let options = HighlightOptions {
        language: Some(lang),
        ignore_illegals: true,
        language_subset: Vec::new(),
        theme: Some(cli_highlight_theme()),
    };
    highlight(code, options)
        .split('\n')
        .map(str::to_string)
        .collect()
}

/// The language a file path's extension names, upstream's
/// `getLanguageFromPath`.
#[must_use]
pub fn get_language_from_path(file_path: &str) -> Option<&'static str> {
    let ext = file_path.split('.').next_back()?.to_lowercase();
    Some(match ext.as_str() {
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "rb" => "ruby",
        "rs" => "rust",
        "go" => "go",
        "java" => "java",
        "kt" => "kotlin",
        "swift" => "swift",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" => "cpp",
        "cs" => "csharp",
        "php" => "php",
        "sh" | "bash" | "zsh" => "bash",
        "fish" => "fish",
        "ps1" => "powershell",
        "sql" => "sql",
        "html" | "htm" => "html",
        "css" => "css",
        "scss" => "scss",
        "sass" => "sass",
        "less" => "less",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "xml" => "xml",
        "md" | "markdown" => "markdown",
        "dockerfile" => "dockerfile",
        "makefile" => "makefile",
        "cmake" => "cmake",
        "lua" => "lua",
        "perl" => "perl",
        "r" => "r",
        "scala" => "scala",
        "clj" => "clojure",
        "ex" | "exs" => "elixir",
        "erl" => "erlang",
        "hs" => "haskell",
        "ml" => "ocaml",
        "vim" => "vim",
        "graphql" => "graphql",
        "proto" => "protobuf",
        "tf" | "hcl" => "hcl",
        _ => return None,
    })
}

/// A per-call foreground color closure for one token.
fn color_fn(token: &'static str) -> pi_tui::components::ColorFn {
    Arc::new(move |text: &str| theme().fg(token, text))
}

/// The Markdown renderer's theme, upstream's `getMarkdownTheme`.
#[must_use]
pub fn get_markdown_theme() -> MarkdownTheme {
    MarkdownTheme {
        heading: color_fn("mdHeading"),
        link: color_fn("mdLink"),
        link_url: color_fn("mdLinkUrl"),
        code: color_fn("mdCode"),
        code_block: color_fn("mdCodeBlock"),
        code_block_border: color_fn("mdCodeBlockBorder"),
        quote: color_fn("mdQuote"),
        quote_border: color_fn("mdQuoteBorder"),
        hr: color_fn("mdHr"),
        list_bullet: color_fn("mdListBullet"),
        bold: Arc::new(|text: &str| bold(text)),
        italic: Arc::new(|text: &str| italic(text)),
        underline: Arc::new(|text: &str| underline(text)),
        strikethrough: Arc::new(|text: &str| strikethrough(text)),
        highlight_code: Some(Arc::new(|code: &str, lang: Option<&str>| {
            highlight_code(code, lang)
        })),
        code_block_indent: None,
    }
}

/// The select list's theme, upstream's `getSelectListTheme`.
#[must_use]
pub fn get_select_list_theme() -> SelectListTheme {
    let token = |color: &'static str| -> Rc<dyn Fn(&str) -> String> {
        Rc::new(move |text: &str| theme().fg(color, text))
    };
    SelectListTheme {
        selected_prefix: token("accent"),
        selected_text: token("accent"),
        description: token("muted"),
        scroll_info: token("muted"),
        no_match: token("muted"),
    }
}

/// The editor's theme, upstream's `getEditorTheme`.
#[must_use]
pub fn get_editor_theme() -> EditorTheme {
    EditorTheme {
        border_color: Rc::new(|text: &str| theme().fg("borderMuted", text)),
        select_list: get_select_list_theme(),
    }
}

/// The settings list's theme, upstream's `getSettingsListTheme`.
#[must_use]
pub fn get_settings_list_theme() -> SettingsListTheme {
    SettingsListTheme {
        label: Rc::new(|text: &str, selected: bool| {
            if selected {
                theme().fg("accent", text)
            } else {
                text.to_string()
            }
        }),
        value: Rc::new(|text: &str, selected: bool| {
            if selected {
                theme().fg("accent", text)
            } else {
                theme().fg("muted", text)
            }
        }),
        description: Rc::new(|text: &str| theme().fg("dim", text)),
        cursor: theme().fg("accent", "→ "),
        hint: Rc::new(|text: &str| theme().fg("dim", text)),
    }
}

/// The thinking-level border color, upstream's
/// `Theme::getThinkingBorderColor`: one dedicated token per level.
#[must_use]
pub fn get_thinking_border_color(level: ThinkingLevel) -> pi_tui::components::ColorFn {
    let token = match level {
        ThinkingLevel::Off => "thinkingOff",
        ThinkingLevel::Minimal => "thinkingMinimal",
        ThinkingLevel::Low => "thinkingLow",
        ThinkingLevel::Medium => "thinkingMedium",
        ThinkingLevel::High => "thinkingHigh",
        ThinkingLevel::Xhigh => "thinkingXhigh",
        ThinkingLevel::Max => "thinkingMax",
    };
    Arc::new(move |text: &str| theme().fg(token, text))
}

/// The bash-mode border color, upstream's `Theme::getBashModeBorderColor`.
#[must_use]
pub fn get_bash_mode_border_color() -> pi_tui::components::ColorFn {
    color_fn("bashMode")
}

#[cfg(test)]
mod tests;
