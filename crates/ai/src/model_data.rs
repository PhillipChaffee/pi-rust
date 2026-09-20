//! Generated model data validation, ported from
//! `packages/ai/scripts/model-data.ts` at commit
//! `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.
//!
//! Per [ADR 0006](docs/adr/0006-generated-model-catalogs-extracted.md) the
//! catalog shards are extracted from the pinned upstream build and committed;
//! this module is the validation half that gates them. The generator itself
//! stays upstream-side, so the provider-id discovery step reads the data
//! directory the committed aggregator module lists, not TypeScript import
//! lines.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;

/// The manifest schema version, upstream's `MODEL_DATA_SCHEMA_VERSION`.
pub const MODEL_DATA_SCHEMA_VERSION: u64 = 3;

/// The manifest file name, upstream's `MODEL_DATA_MANIFEST_FILE`.
pub const MODEL_DATA_MANIFEST_FILE: &str = ".manifest.json";

/// One provider's generated structure: model id to API id, upstream's
/// `ModelDataStructure` value.
pub type ProviderModelApis = BTreeMap<String, String>;

/// The generated model data structure, upstream's `ModelDataStructure`:
/// provider id to (model id, API id).
pub type ModelDataStructure = BTreeMap<String, ProviderModelApis>;

/// The generated data manifest, upstream's `ModelDataManifest`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDataManifest {
    /// The schema version the data was generated against.
    pub schema_version: u64,
    /// ISO timestamp of the generation run.
    pub generated_at: String,
    /// Hash of the canonicalized structure.
    pub structure_hash: String,
    /// Per-file content hashes, keyed by shard file name.
    pub files: BTreeMap<String, String>,
}

/// The failure a model-data operation reports; its display text is the
/// upstream error message.
pub type ModelDataError = Box<dyn std::error::Error + Send + Sync>;

fn error(message: impl Into<String>) -> ModelDataError {
    Box::new(std::io::Error::other(message.into()))
}

/// SHA-256 hex, matching node:crypto's `createHash("sha256")`.
fn sha256(value: &str) -> String {
    let digest = sha2::Sha256::digest(value.as_bytes());
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Exact-allowlist check, upstream's `assertExactModelIds`: both sides
/// deduplicated and sorted.
///
/// # Errors
/// When the sets differ, naming the missing and extra ids.
pub fn assert_exact_model_ids(
    label: &str,
    expected: impl IntoIterator<Item = String>,
    actual: impl IntoIterator<Item = String>,
) -> Result<(), ModelDataError> {
    let expected_ids: BTreeSet<String> = expected.into_iter().collect();
    let actual_ids: BTreeSet<String> = actual.into_iter().collect();
    if expected_ids == actual_ids {
        return Ok(());
    }
    let missing: Vec<String> = expected_ids.difference(&actual_ids).cloned().collect();
    let extra: Vec<String> = actual_ids.difference(&expected_ids).cloned().collect();
    let mut detail = Vec::new();
    if !missing.is_empty() {
        detail.push(format!("missing: {}", missing.join(", ")));
    }
    if !extra.is_empty() {
        detail.push(format!("extra: {}", extra.join(", ")));
    }
    Err(error(format!(
        "{label} model IDs do not match ({})",
        detail.join("; ")
    )))
}

/// Read and parse one JSON file as an object, upstream's `readJsonObject`.
fn read_json_object(path: &Path, description: &str, errors: &mut Vec<String>) -> Option<Value> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(source) => {
            errors.push(format!("{description} is not valid JSON: {source}"));
            return None;
        }
    };
    match serde_json::from_str::<Value>(&content) {
        Ok(value) if value.is_object() => Some(value),
        Ok(_) => {
            errors.push(format!("{description} must contain a JSON object"));
            None
        }
        Err(source) => {
            errors.push(format!("{description} is not valid JSON: {source}"));
            None
        }
    }
}

/// Read one provider shard's structure, upstream's `readProviderStructure`.
#[expect(
    clippy::or_fun_call,
    reason = "the empty-map fallback allocates once per shard, and a const reference cannot"
)]
fn read_provider_structure(
    path: &Path,
    provider_id: &str,
) -> Result<BTreeMap<String, String>, ModelDataError> {
    let mut errors = Vec::new();
    let Some(groups) = read_json_object(path, &format!("{provider_id}.json"), &mut errors) else {
        return Err(error(errors.join("\n")));
    };

    let mut models: BTreeMap<String, String> = BTreeMap::new();
    for (api, value) in groups.as_object().unwrap_or(&serde_json::Map::new()) {
        if !value.is_object() {
            return Err(error(format!(
                "{} API group {api:?} must be an object",
                path.display()
            )));
        }
        for model_id in value.as_object().unwrap_or(&serde_json::Map::new()).keys() {
            if models.contains_key(model_id) {
                return Err(error(format!(
                    "{} contains model {model_id} in more than one API group",
                    path.display()
                )));
            }
            models.insert(model_id.clone(), api.clone());
        }
    }
    if models.is_empty() {
        return Err(error(format!(
            "{} contains no generated model data",
            path.display()
        )));
    }
    Ok(models)
}

/// The provider ids of the generated data directory, upstream's
/// `readModelDataProviderIds` reading `src/models.generated.ts` import lines.
///
/// The Rust aggregator is code, so the committed shards' directory is the
/// source of truth here.
///
/// # Errors
/// No generated provider shards exist in the directory.
pub fn read_model_data_provider_ids(package_root: &Path) -> Result<Vec<String>, ModelDataError> {
    let data_dir = package_root.join("src").join("providers").join("data");
    let mut provider_ids: Vec<String> = std::fs::read_dir(&data_dir)
        .map_err(|source| {
            error(format!(
                "No generated provider imports found in {}: {source}",
                data_dir.display()
            ))
        })?
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(ToOwned::to_owned))
        .filter(|name| {
            #[expect(
                clippy::case_sensitive_file_extension_comparisons,
                reason = "the shards are written with the exact lowercase .json name"
            )]
            let is_json = name.ends_with(".json");
            is_json && name != MODEL_DATA_MANIFEST_FILE
        })
        .map(|name| name.trim_end_matches(".json").to_owned())
        .collect();
    if provider_ids.is_empty() {
        return Err(error(format!(
            "No generated provider imports found in {}",
            data_dir.display()
        )));
    }
    provider_ids.sort();
    Ok(provider_ids)
}

/// The structure of the generated data directory, upstream's
/// `readModelDataStructure`.
///
/// # Errors
/// An unreadable shard, invalid JSON, or a duplicate model across groups.
pub fn read_model_data_structure(
    package_root: &Path,
) -> Result<ModelDataStructure, ModelDataError> {
    let data_dir = package_root.join("src").join("providers").join("data");
    let provider_ids = read_model_data_provider_ids(package_root)?;
    let mut structure = ModelDataStructure::default();
    for provider_id in provider_ids {
        let path = data_dir.join(format!("{provider_id}.json"));
        structure.insert(
            provider_id.clone(),
            read_provider_structure(&path, &provider_id)?,
        );
    }
    Ok(structure)
}

/// The structure hash, upstream's `modelDataStructureHash`: sha256 over the
/// canonicalized `{"provider": {"model": "api"}}` JSON.
#[must_use]
pub fn model_data_structure_hash(structure: &ModelDataStructure) -> String {
    let normalized: BTreeMap<&String, &BTreeMap<String, String>> = structure.iter().collect();
    let json = serde_json::to_string(&normalized).unwrap_or_default();
    sha256(&json)
}

/// Build the manifest for a generated data directory, upstream's
/// `createModelDataManifest`.
#[must_use]
pub fn create_model_data_manifest(
    structure: &ModelDataStructure,
    file_contents: &BTreeMap<String, String>,
    generated_at: &str,
) -> ModelDataManifest {
    let mut files = BTreeMap::new();
    for (file, content) in file_contents {
        files.insert(file.clone(), sha256(content));
    }
    ModelDataManifest {
        schema_version: MODEL_DATA_SCHEMA_VERSION,
        generated_at: generated_at.to_owned(),
        structure_hash: model_data_structure_hash(structure),
        files,
    }
}

/// The missing/extra summary, upstream's `describeSetDifference`.
fn describe_set_difference(expected: &[String], actual: &[String]) -> String {
    let expected_set: BTreeSet<String> = expected.iter().cloned().collect();
    let actual_set: BTreeSet<String> = actual.iter().cloned().collect();
    let missing: Vec<String> = expected
        .iter()
        .filter(|value| !actual_set.contains(*value))
        .cloned()
        .collect();
    let extra: Vec<String> = actual
        .iter()
        .filter(|value| !expected_set.contains(*value))
        .cloned()
        .collect();
    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!("missing: {}", missing.join(", ")));
    }
    if !extra.is_empty() {
        parts.push(format!("extra: {}", extra.join(", ")));
    }
    parts.join("; ")
}

/// Render a JSON field the way upstream's `${JSON.stringify(...)}` templates
/// do, including the `undefined` case for a missing key.
fn json_field(value: Option<&Value>) -> String {
    value.map_or_else(
        || "undefined".to_owned(),
        |inner| serde_json::to_string(inner).unwrap_or_else(|_| "undefined".to_owned()),
    )
}

/// Parse an ISO timestamp the way `Date.parse` accepts the generator's
/// canonical `YYYY-MM-DDTHH:MM:SS.mmmZ` form, `None` when malformed.
fn parse_generated_at(value: &str) -> Option<()> {
    let bytes = value.as_bytes();
    if bytes.len() != 24 {
        return None;
    }
    let digit = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    let char_at = |index: usize| bytes[index] as char;
    (digit(0..4)
        && char_at(4) == '-'
        && digit(5..7)
        && char_at(7) == '-'
        && digit(8..10)
        && char_at(10) == 'T'
        && digit(11..13)
        && char_at(13) == ':'
        && digit(14..16)
        && char_at(16) == ':'
        && digit(17..19)
        && char_at(19) == '.'
        && digit(20..23)
        && char_at(23) == 'Z')
        .then_some(())
}

/// Validate one model value, upstream's `validateModelValue`.
fn validate_model_value(
    value: &Value,
    provider_id: &str,
    model_id: &str,
    expected_api: &str,
    errors: &mut Vec<String>,
) {
    let label = format!("{provider_id}/{model_id}");
    if !value.is_object() {
        errors.push(format!("{label} must be an object"));
        return;
    }
    let field = |name: &str| value.get(name);
    if field("id").and_then(Value::as_str) != Some(model_id) {
        errors.push(format!(
            "{label} has id {}, expected {model_id:?}",
            json_field(value.get("id"))
        ));
    }
    if field("provider").and_then(Value::as_str) != Some(provider_id) {
        errors.push(format!(
            "{label} has provider {}, expected {provider_id:?}",
            json_field(value.get("provider"))
        ));
    }
    if field("api").and_then(Value::as_str) != Some(expected_api) {
        errors.push(format!(
            "{label} has api {}, expected {expected_api:?}",
            json_field(value.get("api"))
        ));
    }
    match value.get("name").and_then(Value::as_str) {
        Some(name) if !name.is_empty() => {}
        _ => errors.push(format!("{label} has no model name")),
    }
    if !value.get("baseUrl").is_some_and(Value::is_string) {
        errors.push(format!("{label} has no baseUrl string"));
    }
    if !value.get("reasoning").is_some_and(Value::is_boolean) {
        errors.push(format!("{label} has no reasoning boolean"));
    }
    let input_valid = value
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            !entries.is_empty()
                && entries
                    .iter()
                    .all(|entry| entry == "text" || entry == "image")
        });
    if !input_valid {
        errors.push(format!("{label} has invalid input modalities"));
    }
    match value.get("contextWindow").and_then(Value::as_f64) {
        Some(window) if window.is_finite() && window > 0.0 => {}
        _ => errors.push(format!("{label} has invalid contextWindow")),
    }
    match value.get("maxTokens").and_then(Value::as_f64) {
        Some(max) if max.is_finite() && max > 0.0 => {}
        _ => errors.push(format!("{label} has invalid maxTokens")),
    }
    let Some(cost) = value.get("cost").and_then(Value::as_object) else {
        errors.push(format!("{label} has invalid cost metadata"));
        return;
    };
    for field in ["input", "output", "cacheRead", "cacheWrite"] {
        let cost_value = cost.get(field).and_then(Value::as_f64);
        match cost_value {
            Some(value) if value.is_finite() => {}
            _ => errors.push(format!("{label} has invalid cost.{field}")),
        }
    }
}

/// Report the collected errors, upstream's `throwValidationErrors`: up to 30
/// visible entries and a count of the rest.
fn throw_validation_errors(errors: &[String]) -> Result<(), ModelDataError> {
    if errors.is_empty() {
        return Ok(());
    }
    let visible: Vec<String> = errors.iter().take(30).cloned().collect();
    let suffix = if errors.len() > visible.len() {
        format!("\n  ... and {} more", errors.len() - visible.len())
    } else {
        String::new()
    };
    Err(error(format!(
        "Invalid generated model data:\n{}{suffix}",
        visible
            .iter()
            .map(|entry| format!("  - {entry}"))
            .collect::<Vec<_>>()
            .join("\n")
    )))
}

/// Validate a generated data directory against its structure, upstream's
/// `validateModelDataDirectory`.
///
/// # Errors
/// Any mismatch between the directory, the manifest, and the structure; the
/// message lists up to 30 errors like upstream's report.
#[expect(
    clippy::too_many_lines,
    reason = "the validation report mirrors upstream's one-pass directory walk"
)]
pub fn validate_model_data_directory(
    structure: &ModelDataStructure,
    data_dir: &Path,
) -> Result<(), ModelDataError> {
    if !data_dir.is_dir() {
        return Err(error(format!(
            "Generated model data directory does not exist: {}",
            data_dir.display()
        )));
    }

    let mut errors: Vec<String> = Vec::new();
    let mut expected_files: Vec<String> = structure
        .keys()
        .map(|provider_id| format!("{provider_id}.json"))
        .collect();
    // Upstream sorts the filename list (`[...].sort()`), not the id list, and
    // the byte orders differ: `-` sorts before `.`, so `google-vertex.json`
    // precedes `google.json`.
    expected_files.sort();
    let mut actual_files: Vec<String> = std::fs::read_dir(data_dir)
        .map_err(|source| error(format!("{}: {source}", data_dir.display())))?
        .flatten()
        .filter_map(|entry| entry.file_name().to_str().map(ToOwned::to_owned))
        .filter(|name| {
            #[expect(
                clippy::case_sensitive_file_extension_comparisons,
                reason = "the shards are written with the exact lowercase .json name"
            )]
            let is_json = name.ends_with(".json");
            is_json && name != MODEL_DATA_MANIFEST_FILE
        })
        .collect();
    actual_files.sort();
    if expected_files != actual_files {
        errors.push(format!(
            "provider data files do not match the generated catalog ({})",
            describe_set_difference(&expected_files, &actual_files)
        ));
    }

    let manifest_path = data_dir.join(MODEL_DATA_MANIFEST_FILE);
    let mut manifest_errors = Vec::new();
    let manifest = read_json_object(&manifest_path, "model data manifest", &mut manifest_errors);
    errors.append(&mut manifest_errors);
    let manifest: Option<ModelDataManifest> = manifest.as_ref().and_then(|value| {
        serde_json::from_value::<ModelDataManifest>(value.clone())
            .inspect_err(|source| {
                errors.push(format!("model data manifest is not valid: {source}"));
            })
            .ok()
    });
    if manifest
        .as_ref()
        .is_none_or(|manifest| manifest.schema_version != MODEL_DATA_SCHEMA_VERSION)
    {
        errors.push(format!(
            "model data schema is {}, expected {MODEL_DATA_SCHEMA_VERSION}",
            manifest.as_ref().map_or_else(
                || "undefined".to_owned(),
                |manifest| serde_json::to_string(&manifest.schema_version).unwrap_or_default()
            )
        ));
    }
    if manifest
        .as_ref()
        .is_none_or(|manifest| parse_generated_at(&manifest.generated_at).is_none())
    {
        errors.push("model data manifest has an invalid generation timestamp".to_owned());
    }
    if let Some(manifest) = &manifest
        && manifest.structure_hash != model_data_structure_hash(structure)
    {
        errors.push("model data generation stamp does not match the generated catalog".to_owned());
    }
    let manifest_files = manifest.as_ref().map(|manifest| &manifest.files);
    match manifest_files {
        None => errors.push("model data manifest has no file hashes".to_owned()),
        Some(manifest_files) => {
            let manifest_names: Vec<String> = manifest_files.keys().cloned().collect();
            if manifest_names != expected_files {
                errors.push(format!(
                    "manifest file hashes do not match provider data files ({})",
                    describe_set_difference(&expected_files, &manifest_names)
                ));
            }
        }
    }

    for (provider_id, expected_models) in structure {
        let filename = format!("{provider_id}.json");
        let path = data_dir.join(&filename);
        if !path.exists() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if let Some(manifest_files) = manifest_files
            && manifest_files.get(&filename).map(String::as_str) != Some(sha256(&content).as_str())
        {
            errors.push(format!("{filename} does not match its manifest hash"));
        }
        let mut shard_errors = Vec::new();
        let Some(groups) = read_json_object(&path, &filename, &mut shard_errors) else {
            errors.append(&mut shard_errors);
            continue;
        };

        let mut actual_models: BTreeMap<String, String> = BTreeMap::new();
        #[expect(
            clippy::or_fun_call,
            reason = "the empty-map fallback allocates once per shard"
        )]
        for (api, value) in groups.as_object().unwrap_or(&serde_json::Map::new()) {
            if !value.is_object() {
                errors.push(format!("{filename} API group {api:?} must be an object"));
                continue;
            }
            for (model_id, model) in value.as_object().unwrap_or(&serde_json::Map::new()) {
                if actual_models.contains_key(model_id) {
                    errors.push(format!(
                        "{provider_id}/{model_id} appears in more than one API group"
                    ));
                    continue;
                }
                actual_models.insert(model_id.clone(), api.clone());
                validate_model_value(model, provider_id, model_id, api, &mut errors);
            }
        }

        let expected_model_ids: Vec<String> = expected_models.keys().cloned().collect();
        let actual_model_ids: Vec<String> = actual_models.keys().cloned().collect();
        if expected_model_ids != actual_model_ids {
            errors.push(format!(
                "{filename} model IDs do not match the generated catalog ({})",
                describe_set_difference(&expected_model_ids, &actual_model_ids)
            ));
        }
        for (model_id, expected_api) in expected_models {
            if let Some(actual_api) = actual_models.get(model_id)
                && actual_api != expected_api
            {
                errors.push(format!(
                    "{provider_id}/{model_id} is grouped under API {actual_api:?}, expected {expected_api:?}"
                ));
            }
        }
    }

    throw_validation_errors(&errors)
}

/// The committed shards' root inside this crate, the port of upstream's
/// `src/providers/data` directory.
#[must_use]
pub fn embedded_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("providers")
        .join("data")
}

/// Validate the committed shards, upstream's `validateGeneratedModelData`.
///
/// # Errors
/// Any inconsistency between the shards and the manifest.
pub fn validate_generated_model_data(package_root: &Path) -> Result<(), ModelDataError> {
    let structure = read_model_data_structure(package_root)?;
    validate_model_data_directory(
        &structure,
        &package_root.join("src").join("providers").join("data"),
    )
}
