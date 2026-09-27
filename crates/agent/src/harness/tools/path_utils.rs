//! Tool-path resolution, ported from upstream `src/harness/tools/path-utils.ts`.
//!
//! The read tool's variant ladder restates upstream's string surgery over
//! the filesystem's `exists` probe: a screenshot-style `10 AM.` filename,
//! macOS-normalization (NFD) differences, and typed apostrophes are the
//! mismatches LLM-supplied paths hit in practice; the first existing variant
//! wins and the lexically resolved path is the fallback.
//!
//! Upstream's `String.prototype.normalize("NFD")` restates over
//! `unicode-normalization`'s `nfd()` iterator. The regex replacements restate
//! as character scans: the matched characters are ASCII and the Unicode-space
//! set is small enough to match by membership.

use unicode_normalization::UnicodeNormalization;

use crate::harness::context::Context;
use crate::harness::types::{ExecutionEnv, FileError};
use crate::types::AgentToolError;

/// The spaces LLM outputs substitute for regular spaces, upstream's
/// `UNICODE_SPACES` regex: NBSP, the U+2000..U+200A run, narrow NBSP, medium
/// math space, and ideographic space.
const fn is_unicode_space(character: char) -> bool {
    matches!(
        character,
        '\u{00A0}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}'
    )
}

/// The narrow no-break space screenshot tools emit before `AM.`/`PM.`, upstream's
/// `NARROW_NO_BREAK_SPACE`.
const NARROW_NO_BREAK_SPACE: char = '\u{202F}';

fn normalize_tool_path(path: &str) -> String {
    let normalized: String = path
        .chars()
        .map(|character| {
            if is_unicode_space(character) {
                ' '
            } else {
                character
            }
        })
        .collect();
    #[allow(
        clippy::option_if_let_else,
        reason = "map_or's default argument moves the path the Some arm's borrow reads; the match is the compiling shape"
    )]
    match normalized.strip_prefix('@') {
        Some(stripped) => stripped.to_owned(),
        None => normalized,
    }
}

/// Replace upstream's `/ (AM|PM)\./gi`: the space before a case-insensitive
/// `AM.`/`PM.` becomes the narrow no-break space, preserving the match's case.
fn replace_ampm_space_separator(path: &str) -> String {
    let characters: Vec<char> = path.chars().collect();
    let mut output = String::with_capacity(path.len());
    let mut index = 0;
    while index < characters.len() {
        let matches = characters[index] == ' '
            && index + 3 < characters.len()
            && matches!(characters[index + 1], 'A' | 'a' | 'P' | 'p')
            && matches!(characters[index + 2], 'M' | 'm')
            && characters[index + 3] == '.';
        if matches {
            output.push(NARROW_NO_BREAK_SPACE);
            output.extend(&characters[index + 1..index + 4]);
            index += 4;
        } else {
            output.push(characters[index]);
            index += 1;
        }
    }
    output
}

/// Resolve a tool-input path to the environment's absolute addressed path,
/// upstream's `resolveToolPath`.
///
/// # Errors
/// The environment's `absolutePath` failure.
pub async fn resolve_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    context: &Context,
) -> Result<String, AgentToolError> {
    let normalized = normalize_tool_path(path);
    Ok(env
        .absolute_path(&normalized, context)
        .await
        .map_err(Box::<FileError>::from)?)
}

/// Resolve a read-tool path, trying the mismatch variants LLM paths hit,
/// upstream's `resolveReadToolPath`.
///
/// # Errors
/// The environment's `absolutePath` failure.
pub async fn resolve_read_tool_path(
    env: &dyn ExecutionEnv,
    path: &str,
    context: &Context,
) -> Result<String, AgentToolError> {
    let resolved = resolve_tool_path(env, path, context).await?;
    let ampm = replace_ampm_space_separator(&resolved);
    let nfd: String = resolved.nfd().collect();
    let apostrophe = resolved.replace('\'', "\u{2019}");
    let nfd_apostrophe = nfd.replace('\'', "\u{2019}");
    let variants = [resolved.clone(), ampm, nfd, apostrophe, nfd_apostrophe];
    for variant in variants {
        if env
            .exists(&variant, context)
            .await
            .map_err(Box::<FileError>::from)?
        {
            return Ok(variant);
        }
    }
    Ok(resolved)
}
