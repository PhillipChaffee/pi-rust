//! BOM handling for decoded text, upstream's `src/utils/text.ts`.

/// A decoded text's leading byte order mark and the remainder.
#[derive(Debug, PartialEq, Eq)]
pub struct SplitBom {
    /// The leading BOM, the U+FEFF character or empty.
    pub bom: String,
    /// The text after the BOM, or the whole input.
    pub text: String,
}

/// Split a leading UTF-8 byte order mark from decoded text.
#[must_use]
pub fn split_bom(content: &str) -> SplitBom {
    let (bom, text) = content.strip_prefix('\u{FEFF}').map_or_else(
        || (String::new(), content.to_string()),
        |text| ('\u{FEFF}'.to_string(), text.to_string()),
    );
    SplitBom { bom, text }
}

/// Remove a leading UTF-8 byte order mark from decoded text.
#[must_use]
pub fn strip_bom(content: &str) -> &str {
    content.strip_prefix('\u{FEFF}').unwrap_or(content)
}
