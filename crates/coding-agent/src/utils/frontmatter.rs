//! `---`-delimited YAML frontmatter, upstream's `src/utils/frontmatter.ts`.
//!
//! The belt's public frontmatter surface: upstream's core skills loader and
//! prompt templates both call into it, so the BOM strip the pi-agent-core
//! harness copy does not carry lives here — the two upstream
//! implementations are deliberate near-duplicates and the port keeps the
//! same split (see `pi_agent_core::harness::fs_scan` for the harness one).

use yaml_rust2::yaml::Hash;
use yaml_rust2::{Yaml, YamlLoader};

use super::text::strip_bom;

/// The frontmatter parse failure, upstream's thrown `Error`.
///
/// Upstream surfaces the npm `yaml` package's thrown message verbatim; the
/// port surfaces yaml-rust2's scanner message for the same malformed
/// inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontmatterError(pub String);

impl std::fmt::Display for FrontmatterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FrontmatterError {}

/// A parsed frontmatter document and the body that follows it, upstream's
/// `{ frontmatter, body }`.
#[derive(Debug, PartialEq, Eq)]
pub struct ParsedFrontmatter {
    /// The parsed frontmatter mapping. Empty, comment-only, or non-object
    /// frontmatter reads as an empty mapping — upstream's `parse(yaml) ?? {}`
    /// followed by property reads that miss on non-objects.
    pub frontmatter: Yaml,
    /// The body text after the closing delimiter, trimmed.
    pub body: String,
}

/// The frontmatter slice upstream's private `extractFrontmatter` scans.
fn extract_frontmatter(content: &str) -> (Option<String>, String) {
    let normalized = normalize_newlines(strip_bom(content));

    if !normalized.starts_with("---") {
        return (None, normalized);
    }

    let Some(end_index) = normalized[3..].find("\n---").map(|at| at + 3) else {
        return (None, normalized);
    };

    // Upstream's `slice(4, endIndex)` clamps a start past the end to an
    // empty string, which the `---\n---…` shape (`endIndex === 3`) hits.
    let yaml_string = if end_index >= 4 {
        normalized[4..end_index].to_string()
    } else {
        String::new()
    };
    let body = normalized
        .get(end_index + 4..)
        .unwrap_or_default()
        .trim()
        .to_string();
    (Some(yaml_string), body)
}

/// Parse the frontmatter and body of a markdown document.
///
/// Line endings normalize to `\n` and a leading BOM strips before
/// scanning. A document that does not start with `---`, or whose closing
/// `\n---` is absent, carries no frontmatter and the whole (normalized)
/// text is the body; the closing delimiter is the first `\n---` after byte
/// 3, so a longer dash run leaves its remainder in the body, upstream's
/// `indexOf`/`slice` pair.
///
/// # Errors
/// A [`FrontmatterError`] when the YAML between the delimiters does not
/// parse, when top-level keys repeat, or when the document contains more
/// than one YAML document — the inputs upstream's `parse` throws on.
pub fn parse_frontmatter(content: &str) -> Result<ParsedFrontmatter, FrontmatterError> {
    let (yaml_string, body) = extract_frontmatter(content);
    let Some(yaml_string) = yaml_string else {
        return Ok(ParsedFrontmatter {
            frontmatter: Yaml::Hash(Hash::new()),
            body,
        });
    };
    let documents = YamlLoader::load_from_str(&yaml_string)
        .map_err(|error| FrontmatterError(error.to_string()))?;
    let frontmatter = match documents.as_slice() {
        [Yaml::Hash(map)] => Yaml::Hash(map.clone()),
        [] | [_] => Yaml::Hash(Hash::new()),
        _ => {
            return Err(FrontmatterError(
                "Source contains multiple documents".to_string(),
            ));
        }
    };
    Ok(ParsedFrontmatter { frontmatter, body })
}

/// The body that follows the frontmatter, upstream's `stripFrontmatter`.
///
/// # Errors
/// The same [`FrontmatterError`] [`parse_frontmatter`] reports.
pub fn strip_frontmatter(content: &str) -> Result<String, FrontmatterError> {
    parse_frontmatter(content).map(|parsed| parsed.body)
}

fn normalize_newlines(value: &str) -> String {
    value.replace("\r\n", "\n").replace('\r', "\n")
}
