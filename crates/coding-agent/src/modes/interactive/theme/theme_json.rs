//! Theme document validation, upstream's
//! `packages/coding-agent/src/modes/interactive/theme/theme-json.ts` at pin
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Upstream validates user-authored themes with typebox but keeps the
//! validator out of `theme.ts` so a presentation that only uses the built-in
//! themes never pays the ~17 MB typebox import. The port has no such import
//! to defer: [`validate_theme_json`] is a hand-written check over the same
//! schema, always on. Its required-token diagnostics keep upstream's wording
//! and ordering.
//!
//! Unknown tokens pass validation, as typebox's `Type.Object` allows
//! additional properties; only the declared tokens are type-checked.

use indexmap::IndexMap;
use serde_json::Value;

use crate::utils::text::strip_bom;

/// One theme color value, upstream's `ThemeColorValue`: a hex string, a
/// variable reference, the empty string for the terminal default, or a
/// 256-color index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThemeColorValue {
    /// A hex color (`#rrggbb`), a variable name, or `""`.
    Text(String),
    /// A 256-color palette index, 0-255.
    Index(u8),
}

/// The explicit HTML-export colors a theme may carry, upstream's `export`
/// section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThemeExportSection {
    /// Upstream `pageBg`.
    pub page_bg: Option<ThemeColorValue>,
    /// Upstream `cardBg`.
    pub card_bg: Option<ThemeColorValue>,
    /// Upstream `infoBg`.
    pub info_bg: Option<ThemeColorValue>,
}

/// A validated theme document, upstream's `ValidatedThemeJson`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedThemeJson {
    /// The theme's display name, upstream `name`; the picker lists this, not
    /// the file name.
    pub name: String,
    /// The variable map color values resolve through, upstream `vars`.
    pub vars: IndexMap<String, ThemeColorValue>,
    /// The color map, upstream `colors`, including any tokens beyond the
    /// schema's.
    pub colors: IndexMap<String, ThemeColorValue>,
    /// The explicit export colors, upstream `export`.
    pub export: Option<ThemeExportSection>,
}

/// The tokens the schema requires under `colors`, in schema order.
const REQUIRED_COLORS: &[&str] = &[
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    "selectedBg",
    "userMessageBg",
    "userMessageText",
    "customMessageBg",
    "customMessageText",
    "customMessageLabel",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "toolTitle",
    "toolOutput",
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "bashMode",
];

/// Validate one theme document, upstream's `validateThemeJson`.
///
/// The error text names the missing required tokens as a group and every
/// other shape failure by JSON path, matching the diagnostics upstream's
/// typebox walk renders.
///
/// # Errors
/// The document is not an object, a declared token is missing or has the
/// wrong type, or `name` contains `/` (reserved for automatic light/dark
/// settings).
#[expect(
    clippy::too_many_lines,
    reason = "the check walks every schema section in one pass, upstream's typebox shape"
)]
pub fn validate_theme_json(label: &str, json: &Value) -> Result<ValidatedThemeJson, String> {
    let Some(object) = json.as_object() else {
        return Err(format!(
            "Invalid theme \"{label}\":\n  - /: Expected an object with a \"colors\" map."
        ));
    };

    let mut other_errors: Vec<String> = Vec::new();

    let name = match object.get("name") {
        Some(Value::String(name)) => name.clone(),
        Some(_) => {
            other_errors.push("  - /name: Expected string".to_string());
            String::new()
        }
        None => {
            other_errors.push("  - /name: Required".to_string());
            String::new()
        }
    };

    let vars = match object.get("vars") {
        None => IndexMap::new(),
        Some(Value::Object(vars)) => {
            let mut parsed = IndexMap::new();
            for (key, value) in vars {
                if let Some(value) = color_value(&format!("/vars/{key}"), value, &mut other_errors)
                {
                    parsed.insert(key.clone(), value);
                }
            }
            parsed
        }
        Some(_) => {
            other_errors.push("  - /vars: Expected object".to_string());
            IndexMap::new()
        }
    };

    let mut missing_colors: Vec<&str> = Vec::new();
    let mut colors = IndexMap::new();
    match object.get("colors") {
        Some(Value::Object(color_map)) => {
            for token in REQUIRED_COLORS {
                match color_map.get(*token) {
                    Some(value) => {
                        if let Some(value) =
                            color_value(&format!("/colors/{token}"), value, &mut other_errors)
                        {
                            colors.insert((*token).to_string(), value);
                        }
                    }
                    None => missing_colors.push(token),
                }
            }
            for (key, value) in color_map {
                if REQUIRED_COLORS.contains(&key.as_str()) {
                    continue;
                }
                if let Some(value) =
                    color_value(&format!("/colors/{key}"), value, &mut other_errors)
                {
                    colors.insert(key.clone(), value);
                }
            }
        }
        Some(_) => {
            other_errors.push("  - /colors: Expected object".to_string());
        }
        None => {
            missing_colors.extend(REQUIRED_COLORS.iter().copied());
        }
    }

    let export = match object.get("export") {
        None => None,
        Some(Value::Object(export)) => {
            let mut section = ThemeExportSection::default();
            for (key, slot) in [
                ("pageBg", &mut section.page_bg),
                ("cardBg", &mut section.card_bg),
                ("infoBg", &mut section.info_bg),
            ] {
                if let Some(value) = export.get(key)
                    && let Some(parsed) =
                        color_value(&format!("/export/{key}"), value, &mut other_errors)
                {
                    *slot = Some(parsed);
                }
            }
            Some(section)
        }
        Some(_) => {
            other_errors.push("  - /export: Expected object".to_string());
            None
        }
    };

    if !missing_colors.is_empty() || !other_errors.is_empty() {
        let mut message = format!("Invalid theme \"{label}\":\n");
        if !missing_colors.is_empty() {
            message.push_str("\nMissing required color tokens:\n");
            let mut sorted = missing_colors.clone();
            sorted.sort_unstable();
            for token in sorted {
                let _ = std::fmt::Write::write_fmt(&mut message, format_args!("  - {token}\n"));
            }
            message.push_str("\nPlease add these colors to your theme's \"colors\" object.");
            message.push_str(
                "\nSee the built-in themes (dark.json, light.json) for reference values.",
            );
        }
        if !other_errors.is_empty() {
            message.push_str("\n\nOther errors:\n");
            message.push_str(&other_errors.join("\n"));
        }
        return Err(message);
    }

    if name.contains('/') {
        return Err(format!(
            "Invalid theme \"{name}\": theme names cannot contain \"/\" because it is reserved for automatic light/dark theme settings."
        ));
    }

    Ok(ValidatedThemeJson {
        name,
        vars,
        colors,
        export,
    })
}

/// Read one color value, recording a shape error under `path` when the JSON
/// is neither a string nor a 256-color index.
fn color_value(path: &str, value: &Value, errors: &mut Vec<String>) -> Option<ThemeColorValue> {
    match value {
        Value::String(text) => Some(ThemeColorValue::Text(text.clone())),
        Value::Number(number) => number
            .as_u64()
            .and_then(|index| u8::try_from(index).ok())
            .map_or_else(
                || {
                    errors.push(format!("  - {path}: Expected a string or an integer 0-255"));
                    None
                },
                |index| Some(ThemeColorValue::Index(index)),
            ),
        _ => {
            errors.push(format!("  - {path}: Expected a string or an integer 0-255"));
            None
        }
    }
}

/// Parse a theme document from file content, upstream's
/// `parseThemeJsonContent`: a BOM is stripped before the JSON parse.
///
/// # Errors
/// The JSON does not parse or [`validate_theme_json`] rejects it.
pub fn parse_theme_json_content(label: &str, content: &str) -> Result<ValidatedThemeJson, String> {
    let json: Value = serde_json::from_str(strip_bom(content))
        .map_err(|error| format!("Failed to parse theme {label}: {error}"))?;
    validate_theme_json(label, &json)
}

#[cfg(test)]
mod tests;
