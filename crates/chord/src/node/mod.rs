//! The node-side manifest and artifact layer, ported from upstream
//! `src/node/*`: the `chord-facets.json` manifest model and its validators,
//! the `chord.facet-bundle` / `chord.facet-bundle-artifact` v2 artifact
//! formats with the `sha256-` integrity prefix, safe file resolution, and
//! the `FacetLoader` trait — the deliberately open seam that replaces
//! upstream's esbuild and `node:vm` execution pipeline with in-process
//! loaders.
