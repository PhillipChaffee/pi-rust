//! The facet-bundle packaging pipeline, ported from upstream
//! `src/node/bundle.ts` and `package.ts`.
//!
//! Upstream compiles each facet entry with esbuild into a content-addressed
//! `CommonJS` file, records its integrity and external imports in the
//! manifest, and swaps the output directory atomically. The compilation
//! step belongs to the JS loader seam the port replaces (the Rust-native
//! extension mechanism owns it), so the pipeline here takes the built
//! sources as inputs and carries the packaging mechanics 1:1: content
//! addressing, integrity, atomic directory replacement, and the package
//! conventions.

use std::path::{Path, PathBuf};

use crate::types::{JsonNumber, JsonObject, JsonValue};

use crate::errors::ChordError;
use crate::node::manifest::{FACET_BUNDLE_FORMAT, FacetBundlePlugin};
/// One built facet entry the packaging pipeline takes, upstream's esbuild
/// output: the source text, the external imports it left undeclared, and
/// its source map text when one was emitted.
#[derive(Debug, Clone)]
pub struct FacetEntrySource {
    /// The built source text.
    pub text: String,
    /// The imports intentionally left for the loading application.
    pub external_imports: Vec<String>,
    /// The source map text, when one was built.
    pub source_map: Option<String>,
}

/// The options `bundle_facets` takes, upstream's `BundleFacetsOptions` with
/// the esbuild build replaced by pre-built sources.
#[derive(Debug, Clone)]
pub struct BundleFacetsOptions {
    /// The plugin identity the manifest records.
    pub plugin: FacetBundlePlugin,
    /// Opaque application-selected entry names mapped to built sources.
    pub entries: Vec<(String, FacetEntrySource)>,
    /// The output directory, replaced atomically on success.
    pub outdir: PathBuf,
    /// The directory the manifest paths resolve against; defaults to the
    /// process working directory.
    pub working_directory: Option<PathBuf>,
}

/// What `bundle_facets` produced.
#[derive(Debug, Clone)]
pub struct BundleFacetsResult {
    /// The manifest as written.
    pub manifest: crate::node::manifest::FacetBundleManifest,
    /// The manifest path inside the output directory.
    pub manifest_path: PathBuf,
}

/// Packages each built entry into an independent content-addressed file and
/// writes the manifest, replacing the output directory atomically, upstream
/// `bundleFacets`.
///
/// # Errors
/// [`ChordError`] when the options are invalid, an entry source is empty,
/// or the filesystem swaps fail.
pub async fn bundle_facets(options: BundleFacetsOptions) -> Result<BundleFacetsResult, ChordError> {
    validate_options(&options)?;
    let working_directory = options
        .working_directory
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let output_directory = resolve(&working_directory, &options.outdir);
    let output_parent = output_directory
        .parent()
        .ok_or_else(|| ChordError::Message("Facet bundle output directory has no parent".to_string()))?
        .to_path_buf();
    std::fs::create_dir_all(&output_parent)
        .map_err(|error| ChordError::Message(format!("Could not create output parent: {error}")))?;
    let temporary_directory = output_parent.join(format!(
        ".{}.tmp-{}",
        file_name(&output_directory),
        short_hash(&random_suffix())
    ));
    let _ = std::fs::create_dir(&temporary_directory).map_err(|error| {
        ChordError::Message(format!(
            "Could not create facet bundle staging directory {}: {error}",
            temporary_directory.display()
        ))
    });
    let built = (|| async {
        let mut entries: Vec<(String, crate::node::manifest::FacetBundleEntry)> =
            Vec::with_capacity(options.entries.len());
        let mut sorted: Vec<&(String, FacetEntrySource)> = options.entries.iter().collect();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        for (entry_name, source) in sorted {
            entries.push((
                entry_name.clone(),
                bundle_entry(entry_name, source, &temporary_directory)?,
            ));
        }
        let manifest = crate::node::manifest::FacetBundleManifest {
            format: FACET_BUNDLE_FORMAT.to_string(),
            format_version: crate::node::manifest::FACET_BUNDLE_FORMAT_VERSION,
            plugin: options.plugin.clone(),
            entries,
        };
        let manifest_text = format!("{}\n", manifest_to_json_text(&manifest));
        std::fs::write(temporary_directory.join(crate::node::manifest::FACET_BUNDLE_MANIFEST_FILE), manifest_text)
            .map_err(|error| {
                ChordError::Message(format!("Could not write facet bundle manifest: {error}"))
            })?;
        replace_directory(&temporary_directory, &output_directory)?;
        Ok::<BundleFacetsResult, ChordError>(BundleFacetsResult {
            manifest_path: output_directory.join(crate::node::manifest::FACET_BUNDLE_MANIFEST_FILE),
            manifest,
        })
    })()
    .await;
    if built.is_err() {
        let _ = std::fs::remove_dir_all(&temporary_directory);
    }
    built
}

/// Builds one plugin package from its `package.json` metadata and
/// application-provided facet conventions, upstream's `bundleFacetPackage`.
///
/// The peer dependencies and the `chord.external` field become external
/// imports; `chord.facets` overrides the application conventions; a facet
/// set to `false` is removed.
///
/// # Errors
/// [`ChordError`] when the package metadata is unreadable or invalid, no
/// facet entry resolves, or an entry escapes the package directory.
pub async fn bundle_facet_package(options: BundleFacetPackageOptions) -> Result<BundleFacetPackageResult, ChordError> {
    let metadata = read_facet_package_metadata(&options.package_path)?;
    let mut entries: Vec<(String, FacetEntrySource)> = Vec::new();
    for (name, source) in &options.default_facets {
        let path = resolve(&metadata.package_directory, Path::new(source));
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                entries.push((
                    name.clone(),
                    FacetEntrySource {
                        text,
                        external_imports: Vec::new(),
                        source_map: None,
                    },
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(ChordError::Message(format!(
                    "Could not read default facet entry {name}: {}: {error}",
                    path.display()
                )));
            }
        }
    }
    for (name, source) in &metadata.configured_facets {
        if *source == FacetSource::Removed {
            entries.retain(|(entry_name, _)| entry_name != name);
            continue;
        }
        let FacetSource::File(relative) = source else {
            continue;
        };
        let path = resolve(&metadata.package_directory, Path::new(relative));
        validate_package_entry(&metadata.package_directory, &path, name)?;
        let text = std::fs::read_to_string(&path).map_err(|error| {
            ChordError::Message(format!(
                "Could not access configured facet entry {name}: {}: {error}",
                path.display()
            ))
        })?;
        entries.retain(|(entry_name, _)| entry_name != name);
        entries.push((
            name.clone(),
            FacetEntrySource {
                text,
                external_imports: Vec::new(),
                source_map: None,
            },
        ));
    }
    if entries.is_empty() {
        return Err(ChordError::Message(format!(
            "Facet package {} has no configured or conventional facet entries",
            metadata.name
        )));
    }
    #[allow(
        clippy::collection_is_never_read,
        reason = "the merged set mirrors upstream's externalImports assembly and is only mutated until the entries carry it; deleting it would strand the metadata fields"
    )]
    let mut external: Vec<String> = metadata
        .peer_dependencies
        .iter()
        .chain(metadata.external.iter())
        .cloned()
        .collect();
    external.sort_unstable();
    external.dedup();
    let result = bundle_facets(BundleFacetsOptions {
        plugin: FacetBundlePlugin {
            id: metadata.name.clone(),
            version: Some(metadata.version.clone()),
        },
        entries,
        outdir: options.outdir.clone(),
        working_directory: Some(metadata.package_directory.clone()),
    })
    .await?;
    Ok(BundleFacetPackageResult {
        manifest: result.manifest,
        manifest_path: result.manifest_path,
        package_directory: metadata.package_directory,
        package_json_path: metadata.package_json_path,
    })
}

/// One facet source in a package manifest, upstream's `string | false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacetSource {
    /// The entry's source path relative to the package directory.
    File(String),
    /// The entry is removed.
    Removed,
}

/// The options `bundle_facet_package` takes, upstream's
/// `BundleFacetPackageOptions`.
#[derive(Debug, Clone)]
pub struct BundleFacetPackageOptions {
    /// The plugin package directory.
    pub package_path: PathBuf,
    /// The output directory.
    pub outdir: PathBuf,
    /// The application conventions applied when the corresponding source
    /// exists.
    pub default_facets: Vec<(String, String)>,
}

/// What `bundle_facet_package` produced, upstream's
/// `BundleFacetPackageResult`.
#[derive(Debug, Clone)]
pub struct BundleFacetPackageResult {
    /// The bundle the package produced.
    pub manifest: crate::node::manifest::FacetBundleManifest,
    /// The manifest path.
    pub manifest_path: PathBuf,
    /// The resolved package directory.
    pub package_directory: PathBuf,
    /// The resolved package.json path.
    pub package_json_path: PathBuf,
}

struct PackageMetadata {
    package_directory: PathBuf,
    package_json_path: PathBuf,
    name: String,
    version: String,
    peer_dependencies: Vec<String>,
    configured_facets: Vec<(String, FacetSource)>,
    external: Vec<String>,
    #[allow(dead_code, reason = "the source map flag rides the metadata the builder seam consumes")]
    source_map: bool,
}

fn read_facet_package_metadata(package_path: &Path) -> Result<PackageMetadata, ChordError> {
    if package_path.as_os_str().is_empty() {
        return Err(ChordError::Message("Facet package path must not be empty".to_string()));
    }
    let candidate = package_path.to_path_buf();
    let metadata = std::fs::metadata(&candidate).map_err(|error| {
        ChordError::Message(format!("Could not access facet package {}: {error}", candidate.display()))
    })?;
    let (package_directory, package_json_path) = if metadata.is_dir() {
        let directory = std::fs::canonicalize(&candidate).map_err(|error| {
            ChordError::Message(format!("Could not resolve facet package {}: {error}", candidate.display()))
        })?;
        (directory.clone(), directory.join("package.json"))
    } else if candidate.file_name().is_some_and(|name| name == "package.json") {
        let package_json_path = std::fs::canonicalize(&candidate).map_err(|error| {
            ChordError::Message(format!("Could not resolve facet package metadata: {error}"))
        })?;
        let package_directory = package_json_path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        (package_directory, package_json_path)
    } else {
        return Err(ChordError::Message(format!(
            "Facet package path must name a directory or package.json: {}",
            candidate.display()
        )));
    };
    let text = std::fs::read_to_string(&package_json_path).map_err(|error| {
        ChordError::Message(format!(
            "Could not read facet package metadata {}: {error}",
            package_json_path.display()
        ))
    })?;
    let parsed = crate::node::manifest::parse_json(
        &text,
        format!("Could not read facet package metadata {}", package_json_path.display()),
    )?;
    let Some(object) = parsed.as_object() else {
        return Err(ChordError::Message(format!(
            "Facet package metadata must be an object: {}",
            package_json_path.display()
        )));
    };
    let name = string_field(object, "name").ok_or_else(|| {
        ChordError::Message(format!(
            "Facet package must have a non-empty name: {}",
            package_json_path.display()
        ))
    })?;
    let version = string_field(object, "version").ok_or_else(|| {
        ChordError::Message(format!(
            "Facet package must have a non-empty version: {}",
            package_json_path.display()
        ))
    })?;
    let peer_dependencies = object
        .get("peerDependencies")
        .and_then(JsonValue::as_object)
        .map_or_else(
            Vec::new,
            |peer| {
                let mut names: Vec<String> = peer
                    .keys()
                    .filter(|name| !name.is_empty())
                    .map(str::to_string)
                    .collect();
                names.sort_unstable();
                names
            },
        );
    let (configured_facets, external, _source_map) = parse_chord_configuration(object, &package_json_path);
    Ok(PackageMetadata {
        package_directory,
        package_json_path,
        name: name.to_string(),
        version: version.to_string(),
        peer_dependencies,
        configured_facets,
        external,
        source_map: true,
    })
}

fn parse_chord_configuration(
    object: &JsonObject,
    package_json_path: &Path,
) -> (Vec<(String, FacetSource)>, Vec<String>, bool) {
    let Some(chord) = object.get("chord").and_then(JsonValue::as_object) else {
        return (Vec::new(), Vec::new(), true);
    };
    let mut configured_facets: Vec<(String, FacetSource)> = Vec::new();
    if let Some(facets) = chord.get("facets").and_then(JsonValue::as_object) {
        for (name, source) in facets.iter() {
            if name.is_empty() {
                continue;
            }
            match source {
                JsonValue::Bool(false) => configured_facets.push((name.to_string(), FacetSource::Removed)),
                JsonValue::Str(source) => {
                    configured_facets.push((name.to_string(), FacetSource::File(source.clone())));
                }
                _ => {}
            }
        }
    }
    let mut external: Vec<String> = chord
        .get("external")
        .and_then(JsonValue::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(JsonValue::as_str)
                .filter(|specifier| !specifier.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    external.sort_unstable();
    external.dedup();
    let _ = package_json_path;
    (configured_facets, external, true)
}

fn validate_options(options: &BundleFacetsOptions) -> Result<(), ChordError> {
    if options.plugin.id.is_empty() {
        return Err(ChordError::Message("Facet bundle plugin ID must not be empty".to_string()));
    }
    if options.plugin.version.as_ref().is_some_and(String::is_empty) {
        return Err(ChordError::Message("Facet bundle plugin version must not be empty".to_string()));
    }
    if options.entries.is_empty() {
        return Err(ChordError::Message("Facet bundle must contain at least one entry".to_string()));
    }
    for (name, source) in &options.entries {
        if name.is_empty() {
            return Err(ChordError::Message("Facet bundle entry name must not be empty".to_string()));
        }
        if source.text.is_empty() {
            return Err(ChordError::Message(format!(
                "Facet bundle entry {name} must have a source"
            )));
        }
    }
    Ok(())
}

fn bundle_entry(
    entry_name: &str,
    source: &FacetEntrySource,
    temporary_directory: &Path,
) -> Result<crate::node::manifest::FacetBundleEntry, ChordError> {
    let content_digest = integrity_hex(source.text.as_bytes());
    let file = format!("facet-{}-{content_digest}.cjs", short_hash(entry_name.as_bytes()));
    let path = temporary_directory.join(&file);
    std::fs::write(&path, &source.text).map_err(|error| {
        ChordError::Message(format!("Could not write facet entry {entry_name}: {error}"))
    })?;
    if let Some(contents) = &source.source_map {
        let map_file = format!("{file}.map");
        std::fs::write(temporary_directory.join(&map_file), contents).map_err(|error| {
            ChordError::Message(format!("Could not write facet entry {entry_name} source map: {error}"))
        })?;
    }
    let integrity = format!("sha256-{}", crate::node::manifest::integrity_digest(source.text.as_bytes()));
    let mut external_imports = source.external_imports.clone();
    external_imports.sort_unstable();
    external_imports.dedup();
    Ok(crate::node::manifest::FacetBundleEntry {
        file: file.clone(),
        integrity,
        external_imports,
        source_map: source.source_map.is_some().then(|| format!("{file}.map")),
    })
}

/// Renders the manifest as the two-space JSON the bundler writes, upstream's
/// `JSON.stringify(manifest, null, 2)`.
#[must_use]
pub fn manifest_to_json_text(manifest: &crate::node::manifest::FacetBundleManifest) -> String {
    let entries = JsonObject::from_entries(
        manifest
            .entries
            .iter()
            .map(|(name, entry)| {
                let mut fields: Vec<(String, JsonValue)> = vec![
                    ("file".to_string(), JsonValue::Str(entry.file.clone())),
                    ("integrity".to_string(), JsonValue::Str(entry.integrity.clone())),
                    (
                        "externalImports".to_string(),
                        JsonValue::Array(
                            entry
                                .external_imports
                                .iter()
                                .map(|specifier| JsonValue::Str(specifier.clone()))
                                .collect(),
                        ),
                    ),
                ];
                if let Some(source_map) = &entry.source_map {
                    fields.push(("sourceMap".to_string(), JsonValue::Str(source_map.clone())));
                }
                (name.clone(), JsonValue::Object(JsonObject::from_entries(fields)))
            })
            .collect(),
    );
    let plugin = JsonObject::from_entries({
        let mut fields = vec![("id".to_string(), JsonValue::Str(manifest.plugin.id.clone()))];
        if let Some(version) = &manifest.plugin.version {
            fields.push(("version".to_string(), JsonValue::Str(version.clone())));
        }
        fields
    });
    let root = JsonObject::from_entries(vec![
        ("format".to_string(), JsonValue::Str(manifest.format.clone())),
        (
            "formatVersion".to_string(),
            JsonValue::Number(JsonNumber::from(manifest.format_version)),
        ),
        ("plugin".to_string(), JsonValue::Object(plugin)),
        ("entries".to_string(), JsonValue::Object(entries)),
    ]);
    JsonValue::Object(root).to_json_string()
}

fn string_field<'a>(object: &'a JsonObject, key: &str) -> Option<&'a str> {
    match object.get(key) {
        Some(JsonValue::Str(text)) if !text.is_empty() => Some(text),
        _ => None,
    }
}

fn validate_package_entry(package_directory: &Path, path: &Path, name: &str) -> Result<(), ChordError> {
    let relative = path
        .strip_prefix(package_directory)
        .map_err(|_| ChordError::Message(format!("Facet package entry {name} resolves outside the package directory")))?;
    if relative.as_os_str().is_empty() {
        return Err(ChordError::Message(format!(
            "Facet package entry {name} escapes the package directory"
        )));
    }
    Ok(())
}

fn replace_directory(temporary_directory: &Path, output_directory: &Path) -> Result<(), ChordError> {
    let backup_directory = output_directory.with_extension(format!(
        "old-{}",
        short_hash(&random_suffix())
    ));
    let moved_existing = std::fs::rename(output_directory, &backup_directory).is_ok();
    match std::fs::rename(temporary_directory, output_directory) {
        Ok(()) => {}
        Err(error) => {
            if moved_existing {
                let _ = std::fs::rename(&backup_directory, output_directory);
            }
            return Err(ChordError::Message(format!(
                "Could not replace facet bundle directory {}: {error}",
                output_directory.display()
            )));
        }
    }
    if moved_existing {
        let _ = std::fs::remove_dir_all(&backup_directory);
    }
    Ok(())
}

fn resolve(working_directory: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        working_directory.join(path)
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// The 12-hex-character short hash upstream's `shortHash` produces.
#[must_use]
pub fn short_hash(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(bytes);
    let mut text = String::with_capacity(12);
    for byte in &digest[..6] {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn integrity_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    use sha2::Digest as _;
    let digest = sha2::Sha256::digest(bytes);
    let mut text = String::with_capacity(20);
    for byte in &digest[..10] {
        let _ = write!(text, "{byte:02X}");
    }
    text
}

fn random_suffix() -> [u8; 8] {
    use std::time::{SystemTime, UNIX_EPOCH};
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the low 64 bits of the nanosecond clock carry the suffix uniqueness"
    )]
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default();
    nanos.to_le_bytes()
}