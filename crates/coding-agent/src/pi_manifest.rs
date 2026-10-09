//! The `pi` package manifest, upstream's `src/core/pi-manifest.ts`.
//!
//! A package's `package.json` may carry a `pi` object naming the extension
//! files, skills, prompts, and themes it ships. The port lands with this
//! ticket rather than the package-manager ticket (#129) because the local
//! package sources the resource loader resolves read it; #129's install
//! machinery consumes the same reader.

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
