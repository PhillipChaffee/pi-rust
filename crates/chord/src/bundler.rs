//! Re-export surface for the manifest and artifact layer.
//!
//! Mirrors upstream's `./bundler` subpath, where upstream exposes
//! `bundleFacets` and `bundleFacetPackage`; the port's packaging contract
//! lives as manifest-level validation in [`crate::node`], so this module
//! surfaces the manifest and artifact APIs that layer owns.

pub use crate::node::{
    BundleFacetPackageOptions, BundleFacetPackageResult, BundleFacetsOptions, BundleFacetsResult,
    FacetBundlePlugin, FacetEntrySource, bundle_facet_package, bundle_facets,
};
