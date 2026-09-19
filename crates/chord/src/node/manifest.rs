//! Facet-bundle manifests and artifacts, ported from upstream
//! `src/node/manifest.ts`.
//!
//! The constants the format carries, the manifest/entry/plugin/artifact
//! shapes, and the versioned validation every read passes. Upstream parses
//! `unknown` objects with `isRecord`/key checks; the port validates its
//! owned [`JsonValue`] tree the same way, and the integrity digest the
//! `sha256-` values carry is verified with the same digest the Node host
//! computed.

use base64::Engine as _;
use sha2::Digest as _;

use crate::errors::ChordError;
use crate::types::{JsonNumber, JsonObject, JsonValue};

/// The manifest format marker.
pub const FACET_BUNDLE_FORMAT: &str = "chord.facet-bundle";
/// The manifest format version this reader understands.
pub const FACET_BUNDLE_FORMAT_VERSION: u64 = 2;
/// The manifest filename inside a bundle directory.
pub const FACET_BUNDLE_MANIFEST_FILE: &str = "chord-facets.json";
/// The artifact format marker.
pub const FACET_BUNDLE_ARTIFACT_FORMAT: &str = "chord.facet-bundle-artifact";
/// The artifact format version this reader understands.
pub const FACET_BUNDLE_ARTIFACT_FORMAT_VERSION: u64 = 2;

/// The `sha256-` subresource-integrity prefix the entries carry.
pub const INTEGRITY_PREFIX: &str = "sha256-";

/// One bundled entry: a content-addressed file, its integrity value, and
/// the imports the loading application must resolve, upstream's
/// `FacetBundleEntry`.
#[derive(Debug, Clone)]
pub struct FacetBundleEntry {
    /// The file name, relative to the manifest.
    pub file: String,
    /// The `sha256-` subresource-integrity value for the file.
    pub integrity: String,
    /// The imports the loading application resolves.
    pub external_imports: Vec<String>,
    /// The source map filename relative to the manifest, when emitted.
    pub source_map: Option<String>,
}

/// The plugin identity a bundle was built for, upstream's
/// `FacetBundlePlugin`.
#[derive(Debug, Clone)]
pub struct FacetBundlePlugin {
    /// The plugin ID.
    pub id: String,
    /// The plugin version, when the builder supplied one.
    pub version: Option<String>,
}

/// One validated facet-bundle manifest.
#[derive(Debug, Clone)]
pub struct FacetBundleManifest {
    /// The format marker, always [`FACET_BUNDLE_FORMAT`].
    pub format: String,
    /// The format version, always [`FACET_BUNDLE_FORMAT_VERSION`].
    pub format_version: u64,
    /// The plugin identity.
    pub plugin: FacetBundlePlugin,
    /// The entries by name, in manifest order.
    pub entries: Vec<(String, FacetBundleEntry)>,
}

impl FacetBundleManifest {
    /// The entry under `name`.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&FacetBundleEntry> {
        self.entries
            .iter()
            .find(|(name_at, _)| name_at == name)
            .map(|(_, entry)| entry)
    }
}

/// One transportable bundle entry: the manifest entry plus the source (and
/// source map) contents, upstream's `FacetBundleArtifact`.
#[derive(Debug, Clone)]
pub struct FacetBundleArtifact {
    /// The artifact format marker.
    pub format: String,
    /// The artifact format version.
    pub format_version: u64,
    /// The plugin identity.
    pub plugin: FacetBundlePlugin,
    /// The entry name inside the manifest.
    pub entry_name: String,
    /// The validated entry.
    pub entry: FacetBundleEntry,
    /// The artifact source text.
    pub source: String,
    /// The source map contents, when the entry declares a source map.
    pub source_map_contents: Option<String>,
}

/// Validates a manifest decoded from JSON, upstream's `validateManifest`.
///
/// # Errors
/// [`ChordError`] describing the first violation, with the reader's path
/// label.
pub fn validate_manifest(value: &JsonValue, path: &str) -> Result<FacetBundleManifest, ChordError> {
    let Some(manifest) = value.as_object() else {
        return Err(manifest_error(path, "Invalid facet bundle manifest format"));
    };
    if manifest.get("format").and_then(JsonValue::as_str) != Some(FACET_BUNDLE_FORMAT) {
        return Err(manifest_error(path, "Invalid facet bundle manifest format"));
    }
    let version = manifest.get("formatVersion").and_then(JsonValue::as_number);
    if version != Some(FACET_BUNDLE_FORMAT_VERSION as f64) {
        return Err(ChordError::Message(format!(
            "Unsupported facet bundle manifest version in {path}: {version:?}"
        )));
    }
    let plugin = manifest
        .get("plugin")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| manifest_error(path, "Facet bundle manifest has an invalid plugin identity"))?;
    let plugin_id = plugin
        .get("id")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| manifest_error(path, "Facet bundle manifest has an invalid plugin identity"))?;
    if plugin_id.is_empty() {
        return Err(manifest_error(path, "Facet bundle manifest has an invalid plugin identity"));
    }
    let version = match plugin.get("version") {
        None => None,
        Some(version) => {
            let version = version.as_str().ok_or_else(|| {
                manifest_error(path, "Facet bundle manifest has an invalid plugin version")
            })?;
            if version.is_empty() {
                return Err(manifest_error(path, "Facet bundle manifest has an invalid plugin version"));
            }
            Some(version.to_string())
        }
    };
    let entries_value = manifest
        .get("entries")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| manifest_error(path, "Facet bundle manifest has no entries"))?;
    if entries_value.is_empty() {
        return Err(manifest_error(path, "Facet bundle manifest has no entries"));
    }
    let mut entries = Vec::with_capacity(entries_value.len());
    for (name, candidate) in entries_value.iter() {
        if name.is_empty() {
            return Err(manifest_error(path, "Facet bundle manifest has an invalid entry"));
        }
        let entry = candidate
            .as_object()
            .ok_or_else(|| manifest_error(path, "Facet bundle manifest has an invalid entry"))?;
        let file = entry
            .get("file")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| manifest_error(path, &format!("Facet bundle entry {name} has no file")))?;
        resolve_bundle_file(file)?;
        let integrity = entry
            .get("integrity")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| manifest_error(path, &format!("Facet bundle entry {name} has no integrity")))?;
        parse_integrity(integrity)?;
        let declared_imports = entry
            .get("externalImports")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                manifest_error(path, &format!("Facet bundle entry {name} has invalid external imports"))
            })?;
        let mut external_imports: Vec<&str> = Vec::with_capacity(declared_imports.len());
        for item in declared_imports {
            let Some(item) = item.as_str() else {
                return Err(manifest_error(
                    path,
                    &format!("Facet bundle entry {name} has invalid external imports"),
                ));
            };
            external_imports.push(item);
        }
        let mut unique = external_imports.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != external_imports.len() {
            return Err(manifest_error(
                path,
                &format!("Facet bundle entry {name} has duplicate external imports"),
            ));
        }
        let source_map = match entry.get("sourceMap") {
            None => None,
            Some(source_map) => {
                let source_map = source_map.as_str().ok_or_else(|| {
                    manifest_error(path, &format!("Facet bundle entry {name} has an invalid source map"))
                })?;
                resolve_bundle_file(source_map)?;
                Some(source_map.to_string())
            }
        };
        entries.push((
            name.to_string(),
            FacetBundleEntry {
                file: file.to_string(),
                integrity: integrity.to_string(),
                external_imports: external_imports.into_iter().map(str::to_string).collect(),
                source_map,
            },
        ));
    }
    Ok(FacetBundleManifest {
        format: FACET_BUNDLE_FORMAT.to_string(),
        format_version: FACET_BUNDLE_FORMAT_VERSION,
        plugin: FacetBundlePlugin {
            id: plugin_id.to_string(),
            version,
        },
        entries,
    })
}

/// Verifies `source` against the entry's integrity value, upstream's
/// `verifySource`.
///
/// # Errors
/// [`ChordError`] when the integrity value is malformed or the digest does
/// not match.
pub fn verify_source(source: &str, entry: &FacetBundleEntry) -> Result<(), ChordError> {
    let expected = parse_integrity(&entry.integrity)?;
    let actual = integrity_digest(source.as_bytes());
    if actual != expected {
        return Err(ChordError::Message(format!(
            "Facet bundle integrity check failed for {}",
            entry.file
        )));
    }
    Ok(())
}

/// The base64 SHA-256 digest the `sha256-` integrity values carry.
#[must_use]
pub fn integrity_digest(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(sha2::Sha256::digest(bytes))
}

/// Splits the integrity value into its digest, upstream's `parseIntegrity`.
///
/// # Errors
/// [`ChordError`] when the value lacks the `sha256-` prefix or digest.
pub fn parse_integrity(integrity: &str) -> Result<String, ChordError> {
    if !integrity.starts_with(INTEGRITY_PREFIX) || integrity.len() == INTEGRITY_PREFIX.len() {
        return Err(ChordError::Message(
            "Facet bundle entry has an invalid SHA-256 integrity value".to_string(),
        ));
    }
    Ok(integrity[INTEGRITY_PREFIX.len()..].to_string())
}

/// Validates one entry filename relative to its manifest, upstream's
/// `resolveBundleFile` shape checks.
///
/// # Errors
/// [`ChordError`] when the name is empty, absolute, or not a plain
/// filename.
pub fn resolve_bundle_file(file: &str) -> Result<(), ChordError> {
    if file.is_empty() || std::path::Path::new(file).is_absolute() || file == "." || file == ".." {
        return Err(ChordError::Message(
            "Facet bundle entry must be a filename relative to its manifest".to_string(),
        ));
    }
    if file.contains('/') || file.contains('\\') {
        return Err(ChordError::Message(
            "Facet bundle entry must be a filename relative to its manifest".to_string(),
        ));
    }
    Ok(())
}

/// Reads and validates a manifest from disk, upstream's
/// `readFacetBundleManifest`.
///
/// # Errors
/// [`ChordError`] when the file cannot be read or the JSON is malformed or
/// invalid.
pub fn read_facet_bundle_manifest(path: &std::path::Path) -> Result<FacetBundleManifest, ChordError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        ChordError::Message(format!("Could not read facet bundle manifest {}: {error}", path.display()))
    })?;
    let parsed = parse_json(&text, format!("Could not read facet bundle manifest {}", path.display()))?;
    validate_manifest(&parsed, &path.to_string_lossy())
}

/// Parses JSON text, wrapping parse failures in the reader's error shape.
///
/// # Errors
/// [`ChordError`] when the text is not valid JSON.
pub fn parse_json(text: &str, context: String) -> Result<JsonValue, ChordError> {
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|error| ChordError::Message(format!("{context}: {error}")))?;
    Ok(json_from_serde(&value))
}

fn json_from_serde(value: &serde_json::Value) -> JsonValue {
    match value {
        serde_json::Value::Null => JsonValue::Null,
        serde_json::Value::Bool(flag) => JsonValue::Bool(*flag),
        serde_json::Value::Number(number) => {
            let number = number.as_f64().unwrap_or_default();
            JsonValue::Number(JsonNumber::new(number).unwrap_or(JsonNumber::new(0.0).expect("zero is finite")))
        }
        serde_json::Value::String(text) => JsonValue::Str(text.clone()),
        serde_json::Value::Array(items) => JsonValue::Array(items.iter().map(json_from_serde).collect()),
        serde_json::Value::Object(entries) => JsonValue::Object(JsonObject::from_entries(
            entries
                .iter()
                .map(|(key, value)| (key.clone(), json_from_serde(value)))
                .collect(),
        )),
    }
}

fn manifest_error(path: &str, message: &str) -> ChordError {
    ChordError::Message(format!("{message} in {path}"))
}