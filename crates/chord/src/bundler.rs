//! Re-export surface for the manifest and artifact layer, mirroring
//! upstream's `./bundler` subpath. Upstream exposes `bundleFacets` and
//! `bundleFacetPackage` here; the port's packaging contract lives as
//! manifest-level validation in [`crate::node`], so this module surfaces
//! the manifest and artifact APIs that layer owns.
