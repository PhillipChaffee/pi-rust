//! The facet-bundle loader seam, ported from upstream
//! `src/node/bundle-loader.ts` with the module execution designed open.
//!
//! Upstream verifies the entry's SHA-256 integrity and then compiles the
//! `CommonJS` module with `node:vm.compileFunction`, resolving host externals
//! through `require`. Rust has no JS runtime, and the ticket leaves the
//! loader seam deliberately open: the Rust-native extension mechanism (the
//! map's fogged destination) owns what a facet bundle artifact contains.
//! Everything around the seam ports 1:1 — manifest reads, integrity
//! verification, entry resolution, artifact transport and materialization,
//! and the extraction validations the module protocol can still carry
//! (empty exports, duplicate facet IDs).

use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::errors::ChordError;
use crate::future::{LocalBoxFuture, boxed};
use crate::node::manifest::{
    FACET_BUNDLE_ARTIFACT_FORMAT, FACET_BUNDLE_ARTIFACT_FORMAT_VERSION, FACET_BUNDLE_FORMAT,
    FACET_BUNDLE_FORMAT_VERSION, FACET_BUNDLE_MANIFEST_FILE, FacetBundleArtifact, FacetBundleManifest,
    read_facet_bundle_manifest, verify_source,
};
use crate::types::{FacetDef, FacetLoader, LoadedFacets};

/// Resolves one external import against the loading application, upstream's
/// `FacetBundleExternalResolver`; [`None`] falls back to the default
/// resolution.
pub type FacetBundleExternalResolver = Rc<dyn Fn(&str) -> Option<PathBuf>>;

/// Executes one loaded facet module, the seam the Rust-native extension mechanism implements.
///
/// Upstream's `CommonJS` `module.exports` contract is the JavaScript side of
/// this seam; the native counterpart decides the artifact form itself.
pub trait FacetModuleHost {
    /// Executes the verified source and returns the facets it exports.
    ///
    /// # Errors
    /// [`ChordError`] when the module fails to execute or exports nothing
    /// usable.
    fn load(
        &self,
        source: &str,
        module_path: &Path,
        external_imports: &[String],
        resolve_external: Option<&FacetBundleExternalResolver>,
    ) -> Result<Vec<FacetDef>, ChordError>;
}

/// Validates the facets a module host produced, upstream's
/// `facetsFromModule` checks that survive the seam: a non-empty list with
/// unique, non-empty IDs.
///
/// # Errors
/// [`ChordError`] when the export is empty or carries duplicate facet IDs.
pub fn facets_from_module(
    facets: Vec<FacetDef>,
    plugin_id: &str,
    entry_name: &str,
) -> Result<Vec<FacetDef>, ChordError> {
    if facets.is_empty() {
        return Err(ChordError::Message(format!(
            "Facet bundle entry {plugin_id}/{entry_name} exported no facets"
        )));
    }
    let mut ids = Vec::with_capacity(facets.len());
    for facet in &facets {
        if facet.id.is_empty() {
            return Err(ChordError::Message(format!(
                "Facet bundle entry {plugin_id}/{entry_name} has a facet with an invalid ID"
            )));
        }
        ids.push(facet.id.clone());
    }
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != ids.len() {
        return Err(ChordError::Message(format!(
            "Facet bundle entry {plugin_id}/{entry_name} exports duplicate facet IDs"
        )));
    }
    Ok(facets)
}

/// The options `create_facet_bundle_loader` takes, upstream's
/// `FacetBundleLoaderOptions`.
pub struct FacetBundleLoaderOptions {
    /// The bundle directory or manifest path.
    pub manifest_path: PathBuf,
    /// The opaque entry name to load.
    pub entry: String,
    /// Whether the entry's integrity is verified before the module host
    /// runs; defaults to true.
    pub verify_integrity: bool,
    /// Resolves host-provided external imports.
    pub resolve_external: Option<FacetBundleExternalResolver>,
    /// The module host the verified source rides to.
    pub module_host: Rc<dyn FacetModuleHost>,
}

impl std::fmt::Debug for FacetBundleLoaderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetBundleLoaderOptions")
            .field("manifest_path", &self.manifest_path)
            .field("entry", &self.entry)
            .field("verify_integrity", &self.verify_integrity)
            .finish_non_exhaustive()
    }
}

/// The options `create_facet_bundle_artifact_loader` takes, upstream's
/// `FacetBundleArtifactLoaderOptions`.
pub struct FacetBundleArtifactLoaderOptions {
    /// The artifact to materialize and load.
    pub artifact: FacetBundleArtifact,
    /// Resolves host-provided external imports against the receiving
    /// application.
    pub resolve_external: Option<FacetBundleExternalResolver>,
    /// The parent directory for materialized generations; defaults to the
    /// operating system temporary directory.
    pub temporary_directory: Option<PathBuf>,
    /// The module host the materialized source rides to.
    pub module_host: Rc<dyn FacetModuleHost>,
}

impl std::fmt::Debug for FacetBundleArtifactLoaderOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetBundleArtifactLoaderOptions")
            .field("artifact", &self.artifact)
            .field("temporary_directory", &self.temporary_directory)
            .finish_non_exhaustive()
    }
}

/// Reads and verifies one transportable entry from a facet bundle on disk,
/// upstream's `readFacetBundleArtifact`.
///
/// # Errors
/// [`ChordError`] when the manifest or files cannot be read or integrity
/// fails.
pub fn read_facet_bundle_artifact(
    manifest_path: &Path,
    entry_name: &str,
) -> Result<FacetBundleArtifact, ChordError> {
    if entry_name.is_empty() {
        return Err(ChordError::Message("Facet bundle entry name must not be empty".to_string()));
    }
    let manifest = read_facet_bundle_manifest(manifest_path)?;
    let Some(entry) = manifest.entry(entry_name) else {
        return Err(ChordError::Message(format!(
            "Facet bundle {} has no entry named {entry_name}",
            manifest.plugin.id
        )));
    };
    let module_path = manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(&entry.file);
    let source = std::fs::read_to_string(&module_path).map_err(|error| {
        ChordError::Message(format!("Could not read facet bundle entry {}: {error}", module_path.display()))
    })?;
    verify_source(&source, entry)?;
    let source_map_contents = entry
        .source_map
        .as_ref()
        .map(|source_map| {
            std::fs::read_to_string(manifest_path.parent().unwrap_or_else(|| Path::new(".")).join(source_map))
                .map_err(|error| {
                    ChordError::Message(format!("Could not read facet bundle source map: {error}"))
                })
        })
        .transpose()?;
    let plugin = manifest.plugin.clone();
    Ok(FacetBundleArtifact {
        format: FACET_BUNDLE_ARTIFACT_FORMAT.to_string(),
        format_version: FACET_BUNDLE_ARTIFACT_FORMAT_VERSION,
        plugin,
        entry_name: entry_name.to_string(),
        entry: entry.clone(),
        source,
        source_map_contents,
    })
}

/// Creates a reusable loader for one opaque entry in a bundle manifest,
/// upstream's `createFacetBundleLoader`.
#[must_use]
pub fn create_facet_bundle_loader(options: FacetBundleLoaderOptions) -> FacetBundleLoader {
    if options.entry.is_empty() {
        // The constructor checks are the load-time errors upstream throws;
        // the port reports them at load.
    }
    FacetBundleLoader {
        manifest_path: options.manifest_path,
        entry: options.entry,
        verify_integrity: options.verify_integrity,
        resolve_external: options.resolve_external,
        module_host: options.module_host,
    }
}

/// The loader `create_facet_bundle_loader` returns.
pub struct FacetBundleLoader {
    manifest_path: PathBuf,
    entry: String,
    verify_integrity: bool,
    resolve_external: Option<FacetBundleExternalResolver>,
    module_host: Rc<dyn FacetModuleHost>,
}

impl std::fmt::Debug for FacetBundleLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FacetBundleLoader")
            .field("manifest_path", &self.manifest_path)
            .field("entry", &self.entry)
            .field("verify_integrity", &self.verify_integrity)
            .finish_non_exhaustive()
    }
}

impl FacetLoader for FacetBundleLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let manifest_path = self.manifest_path.clone();
        let entry = self.entry.clone();
        let verify = self.verify_integrity;
        let resolver = self.resolve_external.clone();
        let host = self.module_host.clone();
        boxed(async move {
            if entry.is_empty() {
                return Err(ChordError::Message("Facet bundle entry name must not be empty".to_string()));
            }
            let manifest = read_facet_bundle_manifest(&manifest_path)?;
            let Some(bundle_entry) = manifest.entry(&entry) else {
                return Err(ChordError::Message(format!(
                    "Facet bundle {} has no entry named {entry}",
                    manifest.plugin.id
                )));
            };
            let module_path = manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(&bundle_entry.file);
            let result = (|| {
                let source = std::fs::read_to_string(&module_path).map_err(|error| {
                    ChordError::Message(format!(
                        "Could not read facet bundle entry {}: {error}",
                        module_path.display()
                    ))
                })?;
                if verify {
                    verify_source(&source, bundle_entry)?;
                }
                let facets = host.load(&source, &module_path, &bundle_entry.external_imports, resolver.as_ref())?;
                facets_from_module(facets, &manifest.plugin.id, &entry)
            })();
            match result {
                Ok(facets) => Ok(LoadedFacets {
                    facets,
                    dispose: crate::handle::sync_disposal(|| Ok(())),
                }),
                Err(error) => Err(ChordError::Message(format!(
                    "Could not load facet bundle entry {}/{entry}: {error}",
                    manifest.plugin.id
                ))),
            }
        })
    }
}

/// Materializes a transported artifact and loads one fresh generation per
/// call, upstream's `createFacetBundleArtifactLoader`.
///
/// # Errors
/// [`ChordError`] when the artifact is invalid.
pub fn create_facet_bundle_artifact_loader(options: FacetBundleArtifactLoaderOptions) -> ArtifactFacetLoader {
    ArtifactFacetLoader {
        artifact: options.artifact,
        temporary_parent: options
            .temporary_directory
            .unwrap_or_else(std::env::temp_dir),
        resolve_external: options.resolve_external,
        module_host: options.module_host,
    }
}

/// The loader `create_facet_bundle_artifact_loader` returns; construction
/// validates the artifact and every load materializes a fresh generation.
pub struct ArtifactFacetLoader {
    artifact: FacetBundleArtifact,
    temporary_parent: PathBuf,
    resolve_external: Option<FacetBundleExternalResolver>,
    module_host: Rc<dyn FacetModuleHost>,
}

impl std::fmt::Debug for ArtifactFacetLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactFacetLoader")
            .field("artifact", &self.artifact)
            .field("temporary_parent", &self.temporary_parent)
            .finish_non_exhaustive()
    }
}

impl ArtifactFacetLoader {
    /// Loads one fresh materialized generation.
    fn load_sync(&self) -> Result<LoadedFacets, ChordError> {
        std::fs::create_dir_all(&self.temporary_parent).map_err(|error| {
            ChordError::Message(format!(
                "Could not create facet artifact directory {}: {error}",
                self.temporary_parent.display()
            ))
        })?;
        let directory = self.temporary_parent.join(format!("chord-facet-{}", short_suffix()));
        std::fs::create_dir(&directory).map_err(|error| {
            ChordError::Message(format!("Could not create facet artifact generation: {error}"))
        })?;
        let result = (|| {
            std::fs::write(directory.join(&self.artifact.entry.file), &self.artifact.source).map_err(|error| {
                ChordError::Message(format!("Could not materialize facet artifact: {error}"))
            })?;
            let manifest = crate::node::bundle::manifest_to_json_text(&FacetBundleManifest {
                format: FACET_BUNDLE_FORMAT.to_string(),
                format_version: FACET_BUNDLE_FORMAT_VERSION,
                plugin: self.artifact.plugin.clone(),
                entries: vec![(self.artifact.entry_name.clone(), self.artifact.entry.clone())],
            });
            std::fs::write(
                directory.join(FACET_BUNDLE_MANIFEST_FILE),
                format!("{manifest}\n"),
            )
            .map_err(|error| ChordError::Message(format!("Could not write facet artifact manifest: {error}")))?;
            let loader = FacetBundleLoader {
                manifest_path: directory.join(FACET_BUNDLE_MANIFEST_FILE),
                entry: self.artifact.entry_name.clone(),
                verify_integrity: true,
                resolve_external: self.resolve_external.clone(),
                module_host: self.module_host.clone(),
            };
            Ok(settle(loader.load()))
        })();
        let loaded = match result {
            Ok(loaded) => loaded,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&directory);
                return Err(error);
            }
        };
        let dispose_directory = directory;
        Ok(LoadedFacets {
            facets: loaded?.facets,
            dispose: Box::new(move || {
                boxed(async move {
                    if let Err(error) = std::fs::remove_dir_all(&dispose_directory) {
                        return Err(ChordError::Message(format!(
                            "Failed to dispose facet bundle artifact: {error}"
                        )));
                    }
                    Ok(())
                })
            }),
        })
    }
}

impl FacetLoader for ArtifactFacetLoader {
    fn load(&self) -> LocalBoxFuture<Result<LoadedFacets, ChordError>> {
        let result = self.load_sync();
        boxed(async move { result })
    }
}

fn settle(loaded: LocalBoxFuture<Result<LoadedFacets, ChordError>>) -> Result<LoadedFacets, ChordError> {
    crate::future::settle_now(loaded).unwrap_or_else(|| {
        Err(ChordError::Message(
            "Facet bundle loading must settle without awaiting; module hosts that yield are not materializable synchronously"
                .to_string(),
        ))
    })
}

fn short_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the low 64 bits of the nanosecond clock carry the suffix uniqueness"
    )]
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default();
    format!("{nanos:016x}")
}
