//! Tool path resolution, ported from upstream `src/core/tools/path-utils.ts`
//! over the path belt's [`crate::utils::paths`] normalizers.
//!
//! The read side tries the resolved path first and then the macOS filename
//! variants — the narrow no-break space before AM/PM in screenshot names,
//! the NFD (decomposed) form macOS stores filenames in, and the curly
//! apostrophe U+2019 macOS uses where users type U+0027.

use regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::utils::paths::{PathInputOptions, normalize_path, resolve_path_with};

/// The narrow no-break space, upstream's `NARROW_NO_BREAK_SPACE`.
const NARROW_NO_BREAK_SPACE: &str = "\u{202F}";

/// The tool-path options every resolver here runs, upstream's
/// `{ normalizeUnicodeSpaces: true, stripAtPrefix: true }`.
fn tool_path_options() -> PathInputOptions {
    PathInputOptions {
        strip_at_prefix: true,
        normalize_unicode_spaces: true,
        ..PathInputOptions::default()
    }
}

/// The macOS screenshot-time suffix, upstream's `/ (AM|PM)\./gi`: the space
/// before AM/PM becomes the narrow no-break space, capture case preserved.
fn try_macos_screenshot_path(file_path: &str) -> String {
    #[expect(
        clippy::expect_used,
        reason = "the grammar is a compile-time constant; a failure is a programming error, not a runtime condition"
    )]
    let pattern = Regex::new(r"(?i) (AM|PM)\.").expect("static pattern");
    pattern
        .replace_all(file_path, format!("{NARROW_NO_BREAK_SPACE}${{1}}."))
        .into_owned()
}

/// The NFD variant: macOS stores filenames in decomposed form, so user input
/// converts to NFD before the existence probe.
fn try_nfd_variant(file_path: &str) -> String {
    file_path.nfd().collect()
}

/// The curly-quote variant: macOS uses U+2019 (right single quotation mark)
/// in screenshot names like "Capture d'écran"; users type U+0027.
fn try_curly_quote_variant(file_path: &str) -> String {
    file_path.replace('\'', "\u{2019}")
}

/// Whether the path exists, upstream's synchronous `fileExists`.
fn file_exists(file_path: &str) -> bool {
    std::path::Path::new(file_path).exists()
}

/// Whether the path exists, upstream's async `pathExists`.
#[must_use]
pub async fn path_exists(file_path: &str) -> bool {
    tokio::fs::metadata(file_path).await.is_ok()
}

/// Normalize a path for tool use, upstream's `expandPath`.
///
/// # Errors
/// [`crate::utils::paths::PathNormalizeError`] when the input is a `file://`
/// URL that does not convert to a local path.
pub fn expand_path(file_path: &str) -> Result<String, crate::utils::paths::PathNormalizeError> {
    normalize_path(file_path, &tool_path_options())
}

/// Resolve a path relative to the given cwd, upstream's `resolveToCwd`.
/// Handles ~ expansion and absolute paths.
///
/// # Errors
/// [`crate::utils::paths::PathNormalizeError`] when the input is a `file://`
/// URL that does not convert to a local path.
pub fn resolve_to_cwd(
    file_path: &str,
    cwd: &str,
) -> Result<String, crate::utils::paths::PathNormalizeError> {
    resolve_path_with(file_path, cwd, &tool_path_options())
}

/// Resolve a read path synchronously, upstream's `resolveReadPath` (its
/// variant ladder is upstream's own duplication).
///
/// # Errors
/// [`crate::utils::paths::PathNormalizeError`] from [`resolve_to_cwd`].
pub fn resolve_read_path(
    file_path: &str,
    cwd: &str,
) -> Result<String, crate::utils::paths::PathNormalizeError> {
    let resolved = resolve_to_cwd(file_path, cwd)?;

    if file_exists(&resolved) {
        return Ok(resolved);
    }

    // Try macOS AM/PM variant (narrow no-break space before AM/PM)
    let am_pm_variant = try_macos_screenshot_path(&resolved);
    if am_pm_variant != resolved && file_exists(&am_pm_variant) {
        return Ok(am_pm_variant);
    }

    // Try NFD variant (macOS stores filenames in NFD form)
    let nfd_variant = try_nfd_variant(&resolved);
    if nfd_variant != resolved && file_exists(&nfd_variant) {
        return Ok(nfd_variant);
    }

    // Try curly quote variant (macOS uses U+2019 in screenshot names)
    let curly_variant = try_curly_quote_variant(&resolved);
    if curly_variant != resolved && file_exists(&curly_variant) {
        return Ok(curly_variant);
    }

    // Try combined NFD + curly quote (for French macOS screenshots like
    // "Capture d'écran")
    let nfd_curly_variant = try_curly_quote_variant(&nfd_variant);
    if nfd_curly_variant != resolved && file_exists(&nfd_curly_variant) {
        return Ok(nfd_curly_variant);
    }

    Ok(resolved)
}

/// Resolve a read path asynchronously, upstream's `resolveReadPathAsync`
/// (its variant ladder is upstream's own duplication).
///
/// # Errors
/// [`crate::utils::paths::PathNormalizeError`] from [`resolve_to_cwd`].
pub async fn resolve_read_path_async(
    file_path: &str,
    cwd: &str,
) -> Result<String, crate::utils::paths::PathNormalizeError> {
    let resolved = resolve_to_cwd(file_path, cwd)?;

    if path_exists(&resolved).await {
        return Ok(resolved);
    }

    // Try macOS AM/PM variant (narrow no-break space before AM/PM)
    let am_pm_variant = try_macos_screenshot_path(&resolved);
    if am_pm_variant != resolved && path_exists(&am_pm_variant).await {
        return Ok(am_pm_variant);
    }

    // Try NFD variant (macOS stores filenames in NFD form)
    let nfd_variant = try_nfd_variant(&resolved);
    if nfd_variant != resolved && path_exists(&nfd_variant).await {
        return Ok(nfd_variant);
    }

    // Try curly quote variant (macOS uses U+2019 in screenshot names)
    let curly_variant = try_curly_quote_variant(&resolved);
    if curly_variant != resolved && path_exists(&curly_variant).await {
        return Ok(curly_variant);
    }

    // Try combined NFD + curly quote (for French macOS screenshots like
    // "Capture d'écran")
    let nfd_curly_variant = try_curly_quote_variant(&nfd_variant);
    if nfd_curly_variant != resolved && path_exists(&nfd_curly_variant).await {
        return Ok(nfd_curly_variant);
    }

    Ok(resolved)
}
