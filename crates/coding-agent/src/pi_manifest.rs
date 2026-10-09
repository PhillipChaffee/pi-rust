//! The `pi` manifest a package's `package.json` may carry, upstream's
//! `src/core/pi-manifest.ts` at pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! The manifest names a package's resource entries per type; the package
//! manager reads it before falling back to convention directories. The
//! `extensions` entries name the extension entry files — TS modules
//! upstream, executable extension binaries in the Rust-native mechanism
//! (ADR 0007) — while the reader itself ports 1:1.

use std::path::Path;

use serde_json::Value;

use crate::utils::text::strip_bom;

/// The resource entries a package declares, upstream's `PiManifest`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PiManifest {
    /// Extension entry files, upstream's `extensions?`.
    pub extensions: Option<Vec<String>>,
    /// Skill directories or files, upstream's `skills?`.
    pub skills: Option<Vec<String>>,
    /// Prompt template files, upstream's `prompts?`.
    pub prompts: Option<Vec<String>>,
    /// Theme JSON files, upstream's `themes?`.
    pub themes: Option<Vec<String>>,
}

const RESOURCE_FIELDS: [&str; 4] = ["extensions", "skills", "prompts", "themes"];

/// Read the `pi` manifest out of a package's `package.json`.
///
/// Returns `None` when the file is missing, is not valid JSON, is not an
/// object, or carries no `pi` object — upstream's `null` on every failure.
/// A field counts only when its entries form an array of strings; a
/// malformed field is silently dropped, not a failure.
pub fn read_pi_manifest(package_json_path: &Path) -> Option<PiManifest> {
    let raw = std::fs::read_to_string(package_json_path).ok()?;
    let pkg: Value = serde_json::from_str(strip_bom(&raw)).ok()?;
    let pi = pkg.get("pi")?;
    if !pi.is_object() {
        return None;
    }

    let mut manifest = PiManifest::default();
    for field in RESOURCE_FIELDS {
        let Some(entries) = pi.get(field).and_then(Value::as_array) else {
            continue;
        };
        if entries.iter().all(Value::is_string) {
            let parsed: Vec<String> = entries
                .iter()
                .map(|entry| entry.as_str().unwrap_or_default().to_string())
                .collect();
            match field {
                "extensions" => manifest.extensions = Some(parsed),
                "skills" => manifest.skills = Some(parsed),
                "prompts" => manifest.prompts = Some(parsed),
                "themes" => manifest.themes = Some(parsed),
                _ => unreachable!("RESOURCE_FIELDS enumerates only these four field names"),
            }
        }
    }
    Some(manifest)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "the unit tests pin manifest reads; an unexpected result panics the test by design"
    )]
    use super::*;

    #[test]
    fn reads_manifest_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("package.json");
        std::fs::write(
            &path,
            r#"{"name":"pkg","pi":{"extensions":["./src/index.ts"],"skills":["./skills"],"prompts":[1,2]}}"#,
        )
        .expect("write");
        let manifest = read_pi_manifest(&path).expect("manifest");
        assert_eq!(
            manifest.extensions,
            Some(vec!["./src/index.ts".to_string()])
        );
        assert_eq!(manifest.skills, Some(vec!["./skills".to_string()]));
        // Non-string entries invalidate the whole field.
        assert_eq!(manifest.prompts, None);
        assert_eq!(manifest.themes, None);
    }

    #[test]
    fn returns_none_without_a_pi_object() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("package.json");
        std::fs::write(&path, r#"{"name":"pkg"}"#).expect("write");
        assert_eq!(read_pi_manifest(&path), None);
    }

    #[test]
    fn returns_none_on_invalid_json_or_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("package.json");
        std::fs::write(&path, "{invalid").expect("write");
        assert_eq!(read_pi_manifest(&path), None);
        assert_eq!(read_pi_manifest(&dir.path().join("absent.json")), None);
    }
}
