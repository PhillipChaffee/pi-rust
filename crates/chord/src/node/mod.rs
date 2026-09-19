//! The Node-side bundle surface, ported from upstream `src/node.ts` and `bundler.ts`.
//!
//! Manifest and artifact contracts with integrity verification, the
//! packaging pipeline over built sources, and the loader seam the
//! Rust-native extension mechanism implements.

pub mod bundle;
pub mod bundle_loader;
pub mod manifest;

pub use bundle::{
    BundleFacetPackageOptions, BundleFacetPackageResult, BundleFacetsOptions, BundleFacetsResult,
    FacetEntrySource, bundle_facet_package, bundle_facets,
};
pub use bundle_loader::{
    ArtifactFacetLoader, FacetBundleArtifactLoaderOptions, FacetBundleExternalResolver,
    FacetBundleLoader, FacetBundleLoaderOptions, FacetModuleHost,
    create_facet_bundle_artifact_loader, create_facet_bundle_loader, facets_from_module,
    read_facet_bundle_artifact,
};
pub use manifest::{
    FACET_BUNDLE_ARTIFACT_FORMAT, FACET_BUNDLE_ARTIFACT_FORMAT_VERSION, FACET_BUNDLE_FORMAT,
    FACET_BUNDLE_FORMAT_VERSION, FACET_BUNDLE_MANIFEST_FILE, FacetBundleArtifact, FacetBundleEntry,
    FacetBundleManifest, FacetBundlePlugin, integrity_digest, parse_integrity,
    read_facet_bundle_manifest, resolve_bundle_file, validate_manifest, verify_source,
};
