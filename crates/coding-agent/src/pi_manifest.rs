//! The `pi` package manifest, upstream's `src/core/pi-manifest.ts`.
//!
//! A package's `package.json` may carry a `pi` object naming the extension
//! files, skills, prompts, and themes it ships. The port lands with the
//! resource-loading ticket (#127) because the local package sources the
//! resource loader resolves read it; the package-manager ticket's (#129)
//! install machinery consumes the same reader. The `extensions` entries
//! name the extension entry files — TS modules upstream, executable
//! extension binaries in the Rust-native mechanism (ADR 0007) — while the
//! reader itself ports 1:1.

use crate::utils::text::strip_bom;

/// A package's declared resources, upstream's `PiManifest`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PiManifest {
    /// The extension entry files, relative to the package root.
    pub extensions: Option<Vec<String>>,
    /// The skill paths, relative to the package root.
    pub skills: Option<Vec<String>>,
    /// The prompt template paths, relative to the package root.
    pub prompts: Option<Vec<String>>,
    /// The theme file paths, relative to the package root.
    pub themes: Option<Vec<String>>,
}

/// The manifest fields this reader scans, upstream's `RESOURCE_FIELDS`.
const RESOURCE_FIELDS: [&str; 4] = ["extensions", "skills", "prompts", "themes"];

/// Read a package's `pi` manifest out of its `package.json`, upstream's
/// `readPiManifest`.
///
/// Returns [`None`] when the file is missing or unparseable, when the JSON
/// is not an object, or when there is no `pi` object. A field reads in
/// only when it is an array of strings, upstream's per-field guard.
#[must_use]
pub fn read_pi_manifest(package_json_path: &str) -> Option<PiManifest> {
    let content = std::fs::read_to_string(package_json_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(strip_bom(&content)).ok()?;
    read_pi_manifest_value(&parsed)
}

/// The manifest reader over an already-parsed `package.json`, the seam the
/// error-reporting layers reuse.
#[must_use]
pub fn read_pi_manifest_value(pkg: &serde_json::Value) -> Option<PiManifest> {
    let pkg = pkg.as_object()?;
    let pi = pkg.get("pi")?.as_object()?;

    let mut manifest = PiManifest::default();
    for field in RESOURCE_FIELDS {
        let entries = pi.get(field).and_then(serde_json::Value::as_array);
        let Some(entries) = entries else {
            continue;
        };
        if !entries.iter().all(serde_json::Value::is_string) {
            continue;
        }
        let strings = entries
            .iter()
            .map(|entry| entry.as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>();
        match field {
            "extensions" => manifest.extensions = Some(strings),
            "skills" => manifest.skills = Some(strings),
            "prompts" => manifest.prompts = Some(strings),
            "themes" => manifest.themes = Some(strings),
            _ => unreachable!("RESOURCE_FIELDS is closed over these four names"),
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
        let manifest = read_pi_manifest(&path.to_string_lossy()).expect("manifest");
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
        assert_eq!(read_pi_manifest(&path.to_string_lossy()), None);
    }

    #[test]
    fn returns_none_on_invalid_json_or_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("package.json");
        std::fs::write(&path, "{invalid").expect("write");
        assert_eq!(read_pi_manifest(&path.to_string_lossy()), None);
        assert_eq!(
            read_pi_manifest(&dir.path().join("absent.json").to_string_lossy()),
            None
        );
    }

    #[test]
    fn reads_a_parsed_value_and_keeps_string_arrays() {
        let parsed: serde_json::Value =
            serde_json::from_str(r#"{"pi":{"themes":["a.json"],"skills":"not-an-array"}}"#)
                .expect("parses");
        let manifest = read_pi_manifest_value(&parsed).expect("manifest");
        assert_eq!(manifest.themes, Some(vec!["a.json".to_string()]));
        assert_eq!(manifest.skills, None);
        // A non-object `pi` and a non-object root both answer None.
        let flat: serde_json::Value = serde_json::from_str(r#"{"pi":[1]}"#).expect("parses");
        assert_eq!(read_pi_manifest_value(&flat), None);
        let scalar: serde_json::Value = serde_json::from_str("3").expect("parses");
        assert_eq!(read_pi_manifest_value(&scalar), None);
    }
}
