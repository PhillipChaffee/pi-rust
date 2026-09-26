//! The filesystem-scanning machinery the skills and prompt-template loaders
//! share, ported from the duplicated helpers in upstream
//! `src/harness/skills.ts` and `src/harness/prompt-templates.ts`.
//!
//! Upstream keeps byte-identical copies of `parseFrontmatter`, `resolveKind`,
//! and the path helpers in both loader files; the port carries them once here
//! because the copy-paste gate measures repeated code shape and one copy is
//! the shape that survives. The two loaders keep their own diagnostic code
//! enums and re-export the shared diagnostic shape under upstream's names.

use std::fmt;

use yaml_rust2::yaml::Hash;
use yaml_rust2::{Yaml, YamlLoader};

use crate::harness::context::Context;
use crate::harness::types::{ExecutionEnv, FileErrorCode, FileInfo, FileKind};

/// Diagnostic severity, upstream's `"warning"` literal in the `type` field.
///
/// Only warnings are emitted today; the field is a value so consumers can
/// switch on it without a format change when error-severity diagnostics
/// appear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// A non-fatal loading problem; the loaded artifact list stays complete.
    Warning,
}

/// One loader diagnostic, upstream's `SkillDiagnostic`/`PromptTemplateDiagnostic`
/// object shape `{ type, code, message, path }`.
///
/// Upstream declares the object twice, once per loader, with the same fields
/// and a different code union; the port declares it once, generic over the
/// code enum, and each loader re-exports it under upstream's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadDiagnostic<C> {
    /// Diagnostic severity; currently only warnings are emitted.
    pub r#type: DiagnosticSeverity,
    /// Stable diagnostic code.
    pub code: C,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Path associated with the diagnostic.
    pub path: String,
}

impl<C> LoadDiagnostic<C> {
    /// Builds one warning diagnostic, the shape every loader push uses.
    pub(crate) fn warning(code: C, message: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            r#type: DiagnosticSeverity::Warning,
            code,
            message: message.into(),
            path: path.into(),
        }
    }
}

/// A diagnostic paired with the source it was loaded for, upstream's
/// `{ ...diagnostic, source }` spread in the sourced loaders.
///
/// Upstream flattens the spread into one object; the port nests the
/// diagnostic under `diagnostic` — the fields and their values are
/// identical, and applications read `sourced.diagnostic.code` where upstream
/// reads `sourced.code`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcedDiagnostic<D, TSource> {
    /// The diagnostic as the plain loader produced it.
    pub diagnostic: D,
    /// The source value the loader input carried.
    pub source: TSource,
}

/// One sourced loader input, upstream's `{ path, source }` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourcedPath<TSource> {
    /// Directory or file path handed to the loader.
    pub path: String,
    /// Application-defined provenance value, preserved exactly.
    pub source: TSource,
}

/// The frontmatter parse failure, upstream's normalized `Error`.
///
/// The message is what the diagnostic's `message` field carries; upstream
/// surfaces the `yaml` package's thrown message verbatim, the port surfaces
/// yaml-rust2's scanner message for the same malformed inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontmatterError(pub String);

impl fmt::Display for FrontmatterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrontmatterError {}

/// A parsed frontmatter document and the body that follows it, upstream's
/// `{ frontmatter, body }`.
#[derive(Debug, PartialEq, Eq)]
pub struct ParsedFrontmatter {
    /// The parsed frontmatter mapping. A missing, empty, or non-object
    /// frontmatter reads as an empty mapping — upstream's `parse(yaml) ?? {}`
    /// followed by property reads that miss on non-objects.
    pub frontmatter: Yaml,
    /// The body text after the closing delimiter, trimmed.
    pub body: String,
}

/// Parse the `---`-delimited YAML frontmatter of a markdown file, upstream's
/// private `parseFrontmatter` (duplicated in `skills.ts` and
/// `prompt-templates.ts`).
///
/// Line endings normalize to `\n` before scanning. A file that does not
/// start with `---`, or whose closing `\n---` is absent, carries no
/// frontmatter and the whole (normalized) text is the body. The closing
/// delimiter is the first `\n---` after byte 3, so a longer run of dashes
/// leaves its remainder in the body, as upstream's `indexOf`/`slice` pair
/// does. Empty YAML and non-object YAML parse as an empty mapping.
///
/// # Errors
/// A [`FrontmatterError`] when the YAML between the delimiters does not
/// parse, when top-level keys repeat, or when the document contains more
/// than one YAML document — the inputs upstream's `parse` throws on.
pub(crate) fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, FrontmatterError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let empty = Yaml::Hash(Hash::new());
    if !normalized.starts_with("---") {
        return Ok(ParsedFrontmatter {
            frontmatter: empty,
            body: normalized,
        });
    }
    let Some(end_index) = normalized[3..].find("\n---").map(|at| at + 3) else {
        return Ok(ParsedFrontmatter {
            frontmatter: empty,
            body: normalized,
        });
    };
    // Upstream's `slice(4, endIndex)` clamps a start past the end to an
    // empty string, which the `endIndex === 3` shape (`---\n---...`) hits.
    let yaml_string = if end_index >= 4 {
        &normalized[4..end_index]
    } else {
        ""
    };
    let body = normalized
        .get(end_index + 4..)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let documents = YamlLoader::load_from_str(yaml_string)
        .map_err(|error| FrontmatterError(error.to_string()))?;
    let frontmatter = match documents.as_slice() {
        [Yaml::Hash(map)] => Yaml::Hash(map.clone()),
        [] | [_] => empty,
        _ => {
            return Err(FrontmatterError(
                "Source contains multiple documents".to_owned(),
            ));
        }
    };
    Ok(ParsedFrontmatter { frontmatter, body })
}

/// Read a string-valued frontmatter field, upstream's
/// `typeof frontmatter.x === "string"` guards.
///
/// Non-string values (numbers, booleans, sequences) and a non-object
/// frontmatter read as absent, exactly as JavaScript property reads miss.
pub(crate) fn frontmatter_string<'a>(frontmatter: &'a Yaml, key: &str) -> Option<&'a str> {
    frontmatter
        .as_hash()
        .and_then(|hash| hash.get(&Yaml::String(key.to_owned())))
        .and_then(Yaml::as_str)
}

/// Whether a frontmatter field is the boolean `true`, upstream's
/// `frontmatter["x"] === true` check. A `"true"` string is not `true`.
pub(crate) fn frontmatter_is_true(frontmatter: &Yaml, key: &str) -> bool {
    frontmatter
        .as_hash()
        .and_then(|hash| hash.get(&Yaml::String(key.to_owned())))
        .is_some_and(|value| matches!(value, Yaml::Boolean(true)))
}

/// Resolve a [`FileInfo`]'s kind to file or directory, upstream's private
/// `resolveKind` (duplicated in `skills.ts` and `prompt-templates.ts`).
///
/// A plain file or directory answers directly; a symlink resolves through
/// [`ExecutionEnv::canonical_path`] and the target's kind. Unresolvable
/// paths answer `None`, silently for `not_found` and with a `file_info_failed`
/// warning (carrying `code`, the failure message, and `info`'s path) for
/// anything else.
pub(crate) async fn resolve_kind<C>(
    env: &dyn ExecutionEnv,
    info: &FileInfo,
    diagnostics: &mut Vec<LoadDiagnostic<C>>,
    code: C,
    context: &Context,
) -> Option<FileKind> {
    if let kind @ (FileKind::File | FileKind::Directory) = info.kind {
        return Some(kind);
    }
    let canonical_path = match env.canonical_path(&info.path, context).await {
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(LoadDiagnostic::warning(
                    code,
                    error.message,
                    info.path.clone(),
                ));
            }
            return None;
        }
        Ok(canonical_path) => canonical_path,
    };
    match env.file_info(&canonical_path, context).await {
        Err(error) => {
            if error.code != FileErrorCode::NotFound {
                diagnostics.push(LoadDiagnostic::warning(
                    code,
                    error.message,
                    info.path.clone(),
                ));
            }
            None
        }
        Ok(target) => match target.kind {
            FileKind::File | FileKind::Directory => Some(target.kind),
            FileKind::Symlink => None,
        },
    }
}

/// The directory part of an addressed path, upstream's private
/// `dirnameEnvPath` (duplicated in `skills.ts` and `prompt-templates.ts`).
///
/// Both separators count and trailing separators are dropped first; a
/// Windows drive root (`C:\`) stays three characters; a path without a
/// separator answers `/`, as upstream's `lastIndexOf` fallback does.
#[must_use]
pub(crate) fn dirname_env_path(path: &str) -> String {
    let normalized = path.trim_end_matches(['/', '\\']);
    // The separators are ASCII, so byte indices cut the same prefixes
    // upstream's UTF-16 indices do.
    match normalized.rfind(['/', '\\']) {
        Some(index) if index == 2 && normalized.as_bytes().get(1) == Some(&b':') => {
            normalized[..3].to_owned()
        }
        Some(index) if index > 0 => normalized[..index].to_owned(),
        _ => "/".to_owned(),
    }
}

/// The path relative to a traversal root, upstream's private
/// `relativeEnvPath` (duplicated in `skills.ts` and `prompt-templates.ts`).
///
/// Backslashes normalize to slashes and trailing separators drop on both
/// sides first. A path at or under the root yields the remainder; anything
/// else loses its leading separators, which keeps the relative shape the
/// ignore matcher requires.
#[must_use]
pub(crate) fn relative_env_path(root: &str, path: &str) -> String {
    let normalized_root = root.replace('\\', "/");
    let normalized_root = normalized_root.trim_end_matches('/');
    let normalized_path = path.replace('\\', "/");
    let normalized_path = normalized_path.trim_end_matches('/');
    if normalized_path == normalized_root {
        return String::new();
    }
    let prefixed = format!("{normalized_root}/");
    if normalized_path.starts_with(&prefixed) {
        normalized_path[prefixed.len()..].to_owned()
    } else {
        normalized_path.trim_start_matches('/').to_owned()
    }
}

/// Sort directory entries by name, upstream's
/// `entries.sort((a, b) => a.name.localeCompare(b.name))`.
///
/// The port sorts by byte order: upstream's `localeCompare` runs ICU
/// collation, which orders ASCII names the same way byte order does for
/// every name the ported suites and their fixtures use.
pub(crate) fn sort_entries_by_name(entries: &mut [FileInfo]) {
    entries.sort_by(|a, b| a.name.cmp(&b.name));
}

#[cfg(test)]
mod tests;
